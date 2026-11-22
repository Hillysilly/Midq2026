// Sparse blossom (exact minimum-weight matching) in C++, called from pipeline.py via ctypes.
//
// This is a line-by-line port of sparse_blossom_match() in pipeline.py; see the comment block
// there for how the algorithm works. Differences from the Python version:
//   * regions live in a std::vector and are referred to by index (Python used objects);
//   * after each shot only the nodes that were touched are reset (Python rebuilt full lists);
//   * whole batches are decoded in one call, so there is no Python work per shot;
//   * each matched pair's observable mask comes from the single edge joining it when that edge
//     is guaranteed to be a shortest path (weight <= 2 * lightest edge), otherwise from a
//     Dijkstra search that never passes through the boundary (same as path_obs()).
//
// Build (pipeline.py does this automatically):
//   g++ -O2 -std=c++17 -shared -static -o blossom.dll blossom.cpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

namespace {

const int NONE = -1;       // no region
const int BOUNDARY = -2;   // "matched to the boundary"
const double TOL = 1e-6;
const long long MAX_EVENTS = 50'000'000;  // safety net against an infinite loop

using Edge = std::pair<int, int>;         // (detector in this region, detector in the other / -1)

// Hot per-region data, kept apart from the vectors in Region so the inner loop of
// schedule_node touches one small array: radius y0 at time t0, and its slope.
struct Hot {
    double y0 = 0.0, t0 = 0.0;
    int slope = 0;                        // +1 growing, -1 shrinking, 0 frozen
};

struct Region {
    std::vector<int> shell;               // nodes grown into while top-level, in order
    int src = -1;                         // the detector, for a single-detector region
    bool blossom = false;
    std::vector<int> children;            // blossom: child regions around the odd cycle
    std::vector<Edge> cycle;              // blossom: cycle[i] = (det in children[i], det in children[i+1])
    int bparent = NONE;                   // blossom containing this region
    int match = NONE;                     // partner region, BOUNDARY or NONE
    Edge medge{-1, -1};
    int tparent = NONE;                   // alternating-tree parent
    Edge tedge{-1, -1};
    std::vector<int> tchildren;
    bool alive = true;
};

// 24 bytes: edge events store the index of the edge in the flat neighbour arrays instead of
// (neighbour, weight); k = -1 marks a region-shrink event (then a is the region).
struct Event {
    double t;
    uint32_t seq;
    int a;
    int k;
    uint32_t ver; // edge events: node a's schedule version when pushed (older versions are dead)
};
// Radix heap: a monotone priority queue (keys popped never decrease, and nothing smaller
// than the last popped key is pushed). Event times are non-negative doubles, whose IEEE bit
// patterns sort exactly like the numbers, so the key is just those 64 bits. An element lives
// in bucket i = index of the highest bit where its key differs from the last popped key, so
// push is O(1) and pop only redistributes one bucket. Pushes are clamped to the last popped
// key (they can only fall below it by floating-point rounding).
//
// Keys never have the sign bit set, so the differing bit is at most bit 62 and buckets 0..63
// suffice; a 64-bit mask of non-empty buckets finds the next one with a single bit scan.
class RadixHeap {
public:
    void push(const Event& e) {
        uint64_t k = key(e.t);
        if (k < last_) k = last_;
        int b = bucket(k);
        buckets_[b].push_back({k, e});
        mask_ |= 1ULL << b;
        ++size_;
    }
    bool empty() const { return size_ == 0; }
    Event pop() {
        if (buckets_[0].empty()) {
            int i = __builtin_ctzll(mask_ & ~1ULL);      // first non-empty bucket above 0
            auto& src = buckets_[i];
            uint64_t m = src[0].first;
            for (const auto& kv : src) m = std::min(m, kv.first);
            last_ = m;
            for (const auto& kv : src) {
                int b = bucket(kv.first);                 // always < i
                buckets_[b].push_back(kv);
                mask_ |= 1ULL << b;
            }
            src.clear();
            mask_ &= ~(1ULL << i);
        }
        Event e = buckets_[0].back().second;
        buckets_[0].pop_back();
        if (buckets_[0].empty()) mask_ &= ~1ULL;
        --size_;
        return e;
    }
    void clear() {                                        // O(1) when already empty
        if (size_) {
            for (auto& b : buckets_) b.clear();
            size_ = 0;
            mask_ = 0;
        }
        last_ = 0;
    }

private:
    static uint64_t key(double t) { uint64_t k; std::memcpy(&k, &t, sizeof k); return k; }
    int bucket(uint64_t k) const { return k == last_ ? 0 : 64 - __builtin_clzll(k ^ last_); }
    std::vector<std::pair<uint64_t, Event>> buckets_[64];
    uint64_t mask_ = 0;
    size_t size_ = 0;
    uint64_t last_ = 0;
};

struct Decoder {
    // Worker copies for sb_decode_batch_mt (made on first use, owned by this decoder).
    std::vector<Decoder*>* workers = nullptr;
    Decoder() = default;
    Decoder(const Decoder&) = default;              // used only to make worker copies
    Decoder& operator=(const Decoder&) = delete;
    ~Decoder() {
        if (workers) { for (Decoder* w : *workers) delete w; delete workers; }
    }
    Decoder* clone() const {
        auto* w = new Decoder(*this);
        w->workers = nullptr;                         // the copy does not own the workers
        return w;
    }

    int n = 0;                                            // detectors; node n = boundary (obs graph only)
    // Matching graph in compressed-sparse-row form: the neighbours of v are
    // adj_to[adj_off[v] .. adj_off[v+1]) with integer weights adj_w (neighbour -1 = boundary).
    std::vector<int> adj_off, adj_to;
    std::vector<double> adj_w;
    // Per CSR edge: observable mask, and whether the edge is guaranteed to be a shortest path
    // (float weight <= 2 * lightest float edge), so pair_obs can read it off directly.
    std::vector<uint64_t> adj_obs;
    std::vector<char> adj_direct;
    std::vector<int> adj_eid;                       // CSR position -> input edge index
    std::vector<int> edge_pos0, edge_pos1;          // input edge -> its CSR positions (pos1 = -1: boundary edge)
    std::vector<std::vector<int>> fnbr;                   // float graph for observable paths (n+1 nodes)
    std::vector<std::vector<double>> fw;
    std::vector<std::vector<uint64_t>> fobs;
    double direct_limit = 0.0;
    // Precomputed per detector (once, in sb_create): shortest distance to the boundary in the
    // integer matching weights, and the observable mask of a shortest boundary path (float weights).
    std::vector<double> bdist;
    std::vector<uint64_t> bobs;
    double int_direct_limit = 0.0;          // integer-weight edges <= this are shortest paths

    // per-shot state
    std::vector<uint32_t> node_ver;   // bumped every time a node is rescheduled
    std::vector<int> top, srcv, trivial_of;
    std::vector<double> base;
    std::vector<int> touched;
    std::vector<Region> R;             // region pool: slots are reused across shots
    std::vector<Hot> H;                // hot data of region i is H[i]
    int nreg = 0;                      // regions in use this shot
    std::vector<int> stk;              // scratch stack for for_each_node
    RadixHeap heap;
    uint32_t seq = 0;
    double now = 0.0;
    long long events = 0;
    std::vector<Edge> pairs;
    std::vector<char> done;

    // ------------------------------------------------------------------ helpers
    double y(int r) const { return H[r].y0 + H[r].slope * (now - H[r].t0); }
    void setslope(int r, int s) { H[r].y0 = y(r); H[r].t0 = now; H[r].slope = s; }
    double local(int v) const { return base[v] + y(top[v]); }
    void cover(int v, int r) { top[v] = r; touched.push_back(v); }

    // A fresh region in a reused slot: vectors are cleared but keep their capacity, so a
    // warmed-up decoder allocates nothing per shot.
    int new_region() {
        if (nreg == (int)R.size()) { R.emplace_back(); H.emplace_back(); }
        H[nreg] = Hot{};
        Region& X = R[nreg];
        X.shell.clear(); X.src = -1; X.blossom = false;
        X.children.clear(); X.cycle.clear();
        X.bparent = NONE; X.match = NONE; X.medge = {-1, -1};
        X.tparent = NONE; X.tedge = {-1, -1}; X.tchildren.clear();
        X.alive = true;
        return nreg++;
    }

    // Visit every node of region r (same order as the old nodes_of list), without allocating.
    // f must not call for_each_node itself or create regions.
    template <class F>
    void for_each_node(int r, F&& f) {
        stk.clear();
        stk.push_back(r);
        while (!stk.empty()) {
            int X = stk.back(); stk.pop_back();
            const Region& RX = R[X];
            if (!RX.blossom) f(RX.src);
            else stk.insert(stk.end(), RX.children.begin(), RX.children.end());
            for (int v : RX.shell) f(v);
        }
    }

    void push_edge(double t, int a, int k, uint32_t ver) { heap.push(Event{t, ++seq, a, k, ver}); }
    void push_shrink(double t, int r) { heap.push(Event{t, ++seq, r, -1, 0}); }

    // Slopes are -1, 0 or +1, so a positive slope (sum) is 1 or 2. Dividing by 1 or 2 is exact
    // in floating point, so "x / s" is replaced by the bit-identical "s == 2 ? x * 0.5 : x".
    static double over(double x, int s) { return s == 2 ? x * 0.5 : x; }

    // When will edge a-b (b = -1: boundary) be fully covered?
    bool edge_time(int a, int b, double w, double& t) const {
        int Ta = top[a];
        if (b == -1) {
            if (Ta < 0 || H[Ta].slope <= 0) return false;
            t = now + (w - local(a));
            return true;
        }
        int Tb = top[b];
        if (Ta < 0 && Tb < 0) return false;
        if (Ta < 0) { std::swap(a, b); std::swap(Ta, Tb); }
        if (Tb < 0) {
            if (H[Ta].slope <= 0) return false;
            t = now + (w - local(a));
            return true;
        }
        if (Ta == Tb) return false;
        int s = H[Ta].slope + H[Tb].slope;
        if (s <= 0) return false;
        t = now + over(w - local(a) - local(b), s);
        return true;
    }

    // One live event per node: the earliest of its edges. Bumping node_ver kills the old one.
    // Every change to an edge's timing reschedules at least one endpoint, and an endpoint's
    // event is re-derived when it turns out stale, so no edge event is ever lost.
    //
    // Same arithmetic as edge_time(v, u, w) (so the check at pop time agrees), but v's region,
    // slope and radius are looked up once instead of once per neighbour.
    void schedule_node(int v) {
        int bk = -1;
        double best = 0.0;
        const int kend = adj_off[v + 1];
        const int Ta = top[v];
        if (Ta >= 0) {
            const int sa = H[Ta].slope;
            const double la = base[v] + y(Ta);
            for (int k = adj_off[v]; k < kend; ++k) {
                const int u = adj_to[k];
                const double w = adj_w[k];
                double t;
                const int Tb = (u == -1) ? -1 : top[u];
                if (Tb < 0) {                              // boundary or empty node
                    if (sa <= 0) continue;
                    t = now + (w - la);
                } else {
                    if (Tb == Ta) continue;
                    const int s = sa + H[Tb].slope;
                    if (s <= 0) continue;
                    t = now + over(w - la - (base[u] + y(Tb)), s);
                }
                if (bk < 0 || t < best) { bk = k; best = t; }
            }
        } else {                                           // v is empty: only growing neighbours
            for (int k = adj_off[v]; k < kend; ++k) {
                const int u = adj_to[k];
                if (u == -1) continue;
                const int Tb = top[u];
                if (Tb < 0 || H[Tb].slope <= 0) continue;
                const double t = now + (adj_w[k] - local(u));
                if (bk < 0 || t < best) { bk = k; best = t; }
            }
        }
        uint32_t ver = ++node_ver[v];
        if (bk >= 0) push_edge(std::max(best, now), v, bk, ver);
    }

    bool shrink_time(int r, double& t) const {
        const Region& X = R[r];
        if (!X.alive || X.bparent != NONE || H[r].slope >= 0) return false;
        t = X.shell.empty() ? now + y(r) : now + local(X.shell.back());
        return true;
    }

    void schedule_region(int r) {
        for_each_node(r, [this](int v) { schedule_node(v); });
        double t;
        if (shrink_time(r, t)) push_shrink(std::max(t, now), r);
    }

    void set_match(int A, int B, Edge e) {
        R[A].match = B; R[A].medge = e;
        if (B != BOUNDARY) { R[B].match = A; R[B].medge = {e.second, e.first}; }
    }

    int root(int r) const { while (R[r].tparent != NONE) r = R[r].tparent; return r; }

    void augment(int A) {
        int cur = A;
        while (R[cur].tparent != NONE) {
            int inner = R[cur].tparent;
            int outer = R[inner].tparent;
            set_match(inner, outer, R[inner].tedge);
            cur = outer;
        }
    }

    std::vector<int> dis_regs, dis_stack;   // scratch for dissolve (no allocation per call)

    // Lazy rescheduling rule. An edge's event time only gets EARLIER when the slope of one of
    // its regions goes UP. When slopes go down (growing -> frozen, frozen -> shrinking) every
    // event of that region can only get later, and the existing (now too early) events are
    // re-derived when they pop. So only regions whose slope rises are rescheduled eagerly.
    void dissolve(int rt) {
        dis_regs.clear();
        dis_stack.clear();
        dis_stack.push_back(rt);
        while (!dis_stack.empty()) {
            int X = dis_stack.back(); dis_stack.pop_back();
            dis_stack.insert(dis_stack.end(), R[X].tchildren.begin(), R[X].tchildren.end());
            R[X].tparent = NONE; R[X].tedge = {-1, -1}; R[X].tchildren.clear();
            if (H[X].slope < 0) dis_regs.push_back(X);   // shrinking -> frozen: slope rises
            setslope(X, 0);
        }
        for (int X : dis_regs) schedule_region(X);   // (does not call dissolve)
    }

    std::vector<int> path_up(int r) const {
        std::vector<int> out{r};
        while (R[r].tparent != NONE) { r = R[r].tparent; out.push_back(r); }
        return out;
    }

    static void replace_child(std::vector<int>& lst, int from, int to) {
        auto it = std::find(lst.begin(), lst.end(), from);
        if (it == lst.end()) throw std::runtime_error("tree child not found");
        *it = to;
    }

    void form_blossom(int A, int B, Edge e) {
        std::vector<int> pa = path_up(A), pb = path_up(B);
        int ib = -1, ia = -1;
        for (int i = 0; i < (int)pb.size() && ib < 0; ++i) {
            auto it = std::find(pa.begin(), pa.end(), pb[i]);
            if (it != pa.end()) { ib = i; ia = int(it - pa.begin()); }
        }
        if (ib < 0) throw std::runtime_error("no common ancestor");
        int lca = pb[ib];
        std::vector<int> down(pa.begin(), pa.begin() + ia + 1);   // A ... lca
        std::reverse(down.begin(), down.end());                    // lca ... A
        std::vector<int> up(pb.begin(), pb.begin() + ib);          // B ... child of lca
        std::vector<int> cyc = down;
        cyc.insert(cyc.end(), up.begin(), up.end());
        std::vector<Edge> edges;
        for (size_t i = 0; i + 1 < down.size(); ++i) {
            Edge te = R[down[i + 1]].tedge;
            edges.push_back({te.second, te.first});
        }
        edges.push_back(e);
        for (int X : up) edges.push_back(R[X].tedge);

        int Bl = new_region();                                     // (no references held across this)
        R[Bl].blossom = true;
        R[Bl].children = cyc;
        R[Bl].cycle = edges;
        H[Bl].slope = 1; H[Bl].y0 = 0.0; H[Bl].t0 = now;
        R[Bl].tparent = R[lca].tparent; R[Bl].tedge = R[lca].tedge;
        if (R[lca].tparent != NONE) replace_child(R[R[lca].tparent].tchildren, lca, Bl);
        R[Bl].match = R[lca].match; R[Bl].medge = R[lca].medge;
        if (R[lca].match != NONE && R[lca].match != BOUNDARY) R[R[lca].match].match = Bl;
        for (int X : cyc)
            for (int ch : R[X].tchildren)
                if (std::find(cyc.begin(), cyc.end(), ch) == cyc.end()) {
                    R[ch].tparent = Bl;
                    R[Bl].tchildren.push_back(ch);
                }
        for (int X : cyc) {
            setslope(X, 0);
            R[X].bparent = Bl;
            R[X].tparent = NONE; R[X].tedge = {-1, -1}; R[X].tchildren.clear();
            double yx = H[X].y0;
            for_each_node(X, [&](int v) { base[v] += yx; top[v] = Bl; });
        }
        schedule_region(Bl);
    }

    void grow_tree(int A, int B, Edge e) {
        int partner = R[B].match;
        R[B].tparent = A; R[B].tedge = {e.second, e.first};
        R[A].tchildren.push_back(B);
        R[partner].tparent = B; R[partner].tedge = R[partner].medge;
        R[B].tchildren = {partner};
        setslope(B, -1);
        setslope(partner, 1);
        double t;                                   // B: frozen -> shrinking (slope falls), so
        if (shrink_time(B, t)) push_shrink(std::max(t, now), B);   // only its shrink event
        schedule_region(partner);                   // partner: frozen -> growing (slope rises)
    }

    void collide(int A, int B, Edge e) {
        if (H[A].slope < H[B].slope) { std::swap(A, B); e = {e.second, e.first}; }
        if (H[B].slope == 1) {
            int ra = root(A), rb = root(B);
            if (ra == rb) {
                form_blossom(A, B, e);
            } else {
                set_match(A, B, e);
                augment(A); augment(B);
                dissolve(ra); dissolve(rb);
            }
        } else if (H[B].slope == 0) {
            if (R[B].match == BOUNDARY) {
                int ra = root(A);
                set_match(A, B, e);
                augment(A);
                dissolve(ra);                       // (B stays frozen: its events are unchanged)
            } else {
                grow_tree(A, B, e);
            }
        } else {
            throw std::runtime_error("collision with a shrinking region");
        }
    }

    void to_boundary(int A, int x) {
        int ra = root(A);
        set_match(A, BOUNDARY, {x, -1});
        augment(A);
        dissolve(ra);
    }

    void shatter(int Bl) {
        int P = R[Bl].tparent;
        Edge tin = R[Bl].tedge;               // (xin, yout)
        int C = R[Bl].tchildren.at(0);
        Edge min_ = R[Bl].medge;              // (yin, zc)
        std::vector<int> ch = R[Bl].children;
        std::vector<Edge> cyc = R[Bl].cycle;
        int m = (int)ch.size();

        auto child_of = [&](int x) {
            int X = trivial_of[x];
            while (R[X].bparent != Bl) X = R[X].bparent;
            return X;
        };
        auto index_in_ch = [&](int X) { return int(std::find(ch.begin(), ch.end(), X) - ch.begin()); };
        auto mod = [m](int k) { return ((k % m) + m) % m; };

        int ip = index_in_ch(child_of(tin.first)), ic = index_in_ch(child_of(min_.first));
        int fwd = mod(ic - ip);
        std::vector<int> path;
        std::vector<Edge> pedges;
        int rest0, nrest;
        if (fwd % 2 == 0) {
            for (int k = 0; k <= fwd; ++k) path.push_back(mod(ip + k));
            for (int k = 0; k < fwd; ++k) pedges.push_back(cyc[mod(ip + k)]);
            rest0 = mod(ic + 1); nrest = m - fwd - 1;
        } else {
            int bk = m - fwd;
            for (int k = 0; k <= bk; ++k) path.push_back(mod(ip - k));
            for (int k = 0; k < bk; ++k) {
                Edge c = cyc[mod(ip - k - 1)];
                pedges.push_back({c.second, c.first});
            }
            rest0 = mod(ip + 1); nrest = m - bk - 1;
        }
        for (int X : ch) {
            R[X].bparent = NONE;
            double yx = y(X);
            H[X].y0 = yx; H[X].t0 = now;
            for_each_node(X, [&](int v) { base[v] -= yx; top[v] = X; });
        }
        std::vector<int> q;
        for (int i : path) q.push_back(ch[i]);
        R[q[0]].tparent = P; R[q[0]].tedge = tin;
        replace_child(R[P].tchildren, Bl, q[0]);
        for (size_t k = 0; k + 1 < q.size(); ++k) {
            Edge pe = pedges[k];
            R[q[k + 1]].tparent = q[k]; R[q[k + 1]].tedge = {pe.second, pe.first};
            R[q[k]].tchildren = {q[k + 1]};
            if (k % 2 == 0) set_match(q[k], q[k + 1], pe);
        }
        int last = q.back();
        R[last].tchildren = {C};
        set_match(last, C, min_);
        R[C].tparent = last; R[C].tedge = {min_.second, min_.first};
        for (size_t k = 0; k < q.size(); ++k) setslope(q[k], k % 2 == 0 ? -1 : 1);
        for (int k = 0; k < nrest; k += 2) {
            int i1 = mod(rest0 + k), i2 = mod(rest0 + k + 1);
            int X = ch[i1], Y = ch[i2];
            set_match(X, Y, cyc[i1]);
            for (int Z : {X, Y}) {
                R[Z].tparent = NONE; R[Z].tedge = {-1, -1}; R[Z].tchildren.clear();
                setslope(Z, 0);
            }
        }
        R[Bl].alive = false;
        for (int X : ch) schedule_region(X);
    }

    void trivial_zero(int r) {
        int s = R[r].src;
        int P = R[r].tparent, C = R[r].tchildren.at(0);
        int p = -1, c = -1;
        for (int k = adj_off[s], e = adj_off[s + 1]; k < e; ++k) {
            int u = adj_to[k];
            double w = adj_w[k];
            if (u == -1 || top[u] < 0) continue;
            if (p < 0 && top[u] == P && local(u) >= w - TOL) p = u;
            if (c < 0 && top[u] == C && local(u) >= w - TOL) c = u;
        }
        if (p < 0 || c < 0) throw std::runtime_error("degenerate region with no touching neighbours");
        collide(P, C, {srcv[p], srcv[c]});
    }

    // ------------------------------------------------------------------ matching
    void match(const std::vector<int>& defects) {
        pairs.clear();
        now = 0.0;
        events = 0;
        seq = 0;
        for (int x : defects) {
            int r = new_region();
            R[r].src = x; H[r].slope = 1;
            trivial_of[x] = r;
            cover(x, r); base[x] = 0.0; srcv[x] = x;
        }
        for (int x : defects) schedule_node(x);

        while (!heap.empty()) {
            Event ev = heap.pop();
            if (++events > MAX_EVENTS) throw std::runtime_error("too many events");
            if (ev.k >= 0) {
                if (ev.ver != node_ver[ev.a]) continue;          // superseded: a newer event exists
                const int ev_u = adj_to[ev.k];
                double tt;
                if (!edge_time(ev.a, ev_u, adj_w[ev.k], tt) || std::fabs(tt - ev.t) > TOL) {
                    schedule_node(ev.a);                           // out of date: re-derive it
                    continue;
                }
                now = std::max(now, ev.t);
                const uint32_t ver_at_pop = node_ver[ev.a];
                struct Resched {                                   // the node's event was used up:
                    Decoder* D; int v; uint32_t ver;              // give it a new one unless the
                    ~Resched() { if (D->node_ver[v] == ver) D->schedule_node(v); }  // handling did
                } resched{this, ev.a, ver_at_pop};
                int a = ev.a, u = ev_u;
                int Ta = top[a];
                if (u == -1) {
                    to_boundary(Ta, srcv[a]);
                } else if (Ta < 0 || top[u] < 0) {       // grew into an empty node
                    if (Ta < 0) { std::swap(a, u); Ta = top[a]; }
                    cover(u, Ta);
                    base[u] = -y(Ta);
                    srcv[u] = srcv[a];
                    R[Ta].shell.push_back(u);
                    schedule_node(u);
                } else {
                    collide(Ta, top[u], {srcv[a], srcv[u]});
                }
            } else {
                int r = ev.a;
                double tt;
                if (!shrink_time(r, tt) || std::fabs(tt - ev.t) > TOL) continue;
                now = std::max(now, ev.t);
                if (!R[r].shell.empty()) {
                    int v = R[r].shell.back(); R[r].shell.pop_back();
                    top[v] = -1; srcv[v] = -1;
                    for (int k = adj_off[v], e = adj_off[v + 1]; k < e; ++k) {
                        int u = adj_to[k];
                        if (u != -1 && top[u] >= 0) schedule_node(u);
                    }
                    double t2;
                    if (shrink_time(r, t2)) push_shrink(std::max(t2, now), r);
                } else if (R[r].blossom) {
                    shatter(r);
                } else {
                    trivial_zero(r);
                }
            }
        }

        // expand blossoms into detector pairs
        // Top-level regions in order of first appearance, one pass using the done flags.
        done.assign(nreg, 0);
        for (int x : defects) {
            int T = trivial_of[x];
            while (R[T].bparent != NONE) T = R[T].bparent;
            if (done[T]) continue;
            if (R[T].match == NONE) throw std::runtime_error("unmatched region at the end");
            Edge me = R[T].medge;
            pairs.push_back(me);
            expand(T, me.first);
            done[T] = 1;
            if (R[T].match != BOUNDARY) { expand(R[T].match, me.second); done[R[T].match] = 1; }
        }
    }

    // Pair up the children of blossom Rr given that detector x (inside it) is matched outside.
    void expand(int Rr, int x) {
        while (R[Rr].blossom) {
            int m = (int)R[Rr].children.size();
            int X = trivial_of[x];
            while (R[X].bparent != Rr) X = R[X].bparent;
            const std::vector<int>& ch = R[Rr].children;
            int j = int(std::find(ch.begin(), ch.end(), X) - ch.begin());
            for (int k = 1; k < m; k += 2) {
                int i1 = (j + k) % m, i2 = (j + k + 1) % m;
                Edge c = R[Rr].cycle[i1];
                pairs.push_back(c);
                expand(R[Rr].children[i1], c.first);
                expand(R[Rr].children[i2], c.second);
            }
            Rr = X;
        }
    }

    void reset(const std::vector<int>& defects) {
        for (int v : touched) { top[v] = -1; base[v] = 0.0; srcv[v] = -1; }
        touched.clear();
        for (int x : defects) trivial_of[x] = -1;
        nreg = 0;
        heap.clear();
    }

    // ------------------------------------------------------------------ observables
    // Scratch for the pair_obs fallback search, sized once; only touched entries are reset.
    std::vector<double> po_dist;
    std::vector<uint64_t> po_obs;
    std::vector<char> po_fin;
    std::vector<int> po_touched;
    std::vector<std::pair<double, int>> po_heap;

    uint64_t pair_obs(int x, int target) {
        if (target == n) return bobs[x];                 // boundary: precomputed shortest path
        for (int k = adj_off[x], e = adj_off[x + 1]; k < e; ++k)
            if (adj_to[k] == target) {
                if (adj_direct[k]) return adj_obs[k];
                break;
            }
        // Dijkstra from x to target; the boundary (node n) is a sink, never passed through.
        if (po_dist.empty()) {
            po_dist.assign(n + 1, std::numeric_limits<double>::infinity());
            po_obs.assign(n + 1, 0);
            po_fin.assign(n + 1, 0);
        }
        auto cmp = std::greater<std::pair<double, int>>();
        po_heap.clear();
        po_dist[x] = 0.0; po_obs[x] = 0; po_touched.push_back(x);
        po_heap.push_back({0.0, x});
        uint64_t result = 0;
        bool found = false;
        while (!po_heap.empty()) {
            std::pop_heap(po_heap.begin(), po_heap.end(), cmp);
            auto [d, v] = po_heap.back(); po_heap.pop_back();
            if (po_fin[v]) continue;
            po_fin[v] = 1;
            if (v == target) { result = po_obs[v]; found = true; break; }
            if (v == n) continue;
            for (size_t k = 0; k < fnbr[v].size(); ++k) {
                int u = fnbr[v][k];
                double nd = d + fw[v][k];
                if (!po_fin[u] && nd < po_dist[u]) {
                    if (po_dist[u] == std::numeric_limits<double>::infinity()) po_touched.push_back(u);
                    po_dist[u] = nd; po_obs[u] = po_obs[v] ^ fobs[v][k];
                    po_heap.push_back({nd, u});
                    std::push_heap(po_heap.begin(), po_heap.end(), cmp);
                }
            }
        }
        for (int v : po_touched) { po_dist[v] = std::numeric_limits<double>::infinity(); po_fin[v] = 0; }
        po_touched.clear();
        if (!found) throw std::runtime_error("no path");
        return result;
    }

    // Exact shortcuts for shots with one or two lit detectors.
    //  * one: it can only be matched to the boundary.
    //  * two joined by a shortest-path edge: the only perfect matchings are {x-y} and
    //    {x-boundary, y-boundary}; compare their (integer) weights. Exact ties go to the
    //    general algorithm so tie-breaking stays identical to it.
    bool shortcut(const std::vector<int>& defects, uint64_t& mask) {
        if (defects.size() == 1) {
            if (bdist[defects[0]] == std::numeric_limits<double>::infinity()) return false;
            mask = bobs[defects[0]];
            return true;
        }
        if (defects.size() == 2) {
            int x = defects[0], y = defects[1];
            for (int k = adj_off[x], e = adj_off[x + 1]; k < e; ++k) {
                if (adj_to[k] != y) continue;
                double w = adj_w[k];
                if (w > int_direct_limit) return false;
                double wb = bdist[x] + bdist[y];
                if (w < wb) { mask = pair_obs(x, y); return true; }
                if (w > wb) { mask = bobs[x] ^ bobs[y]; return true; }
                return false;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ correlated matching
    // PyMatching-style two-pass correlated matching (enable_correlations=True):
    //   pass 1: ordinary matching; collect the edges used an odd number of times by the
    //           matched shortest paths;
    //   reweight: every used edge e lowers each correlated edge f to the implied weight
    //           w(f|e) = -log-odds of min(0.5, P(e and f) / P(e)) if that is lower (weights
    //           only ever decrease; several rules on one edge keep the minimum);
    //   pass 2: ordinary matching on the reweighted graph; its paths give the observables;
    //   undo:   restore the weights.
    // Weights change from shot to shot here, so nothing precomputed from the fixed weights
    // (boundary paths, the global direct-edge rule, small-shot shortcuts) is used.
    bool correlated = false;
    int m = 0;                                      // number of input edges
    std::vector<int> rule_off, rule_f;              // implied-weight rules of edge e:
    std::vector<double> rule_w;                     //   rule_f/rule_w[rule_off[e] .. rule_off[e+1])
    double base_minB = std::numeric_limits<double>::infinity();   // lightest boundary edge
    double cur_minB = std::numeric_limits<double>::infinity();
    std::vector<char> eflip;                        // per edge: used an odd number of times
    std::vector<char> eseen;                        // per edge: already listed in flipped
    std::vector<int> flipped;                       // edges whose parity was touched
    std::vector<std::pair<int, double>> saved;      // (edge, previous weight) for undo
    std::vector<int> po_from;                       // Dijkstra: CSR position of the parent edge
    std::vector<int> adj_src;                       // CSR position -> node it leaves from
    // Two lightest non-boundary neighbours of each node in the BASE weights, valid while none
    // of the node's edges is reweighted (ndirty[v] counts its currently reweighted edges).
    std::vector<double> top1w, top2w;
    std::vector<int> top1nbr;
    std::vector<int> ndirty;

    double edge_weight(int e) const { return adj_w[edge_pos0[e]]; }
    void mark_dirty(int e, int delta) {
        ndirty[adj_src[edge_pos0[e]]] += delta;
        if (edge_pos1[e] >= 0) ndirty[adj_src[edge_pos1[e]]] += delta;
    }
    // Lightest current non-boundary edge at y other than the one to x.
    double min_other_fast(int y, int x) const {
        if (ndirty[y]) return min_other(y, x);
        return top1nbr[y] != x ? top1w[y] : top2w[y];
    }
    void set_edge_weight(int e, double w) {
        adj_w[edge_pos0[e]] = w;
        if (edge_pos1[e] >= 0) adj_w[edge_pos1[e]] = w;
    }
    void flip(int e) {
        if (!eseen[e]) { eseen[e] = 1; flipped.push_back(e); }
        eflip[e] ^= 1;
    }

    // Lightest current non-boundary edge at x, ignoring edges to `skip`.
    double min_other(int x, int skip) const {
        double mn = std::numeric_limits<double>::infinity();
        for (int k = adj_off[x], e = adj_off[x + 1]; k < e; ++k)
            if (adj_to[k] != -1 && adj_to[k] != skip) mn = std::min(mn, adj_w[k]);
        return mn;
    }

    // Observable mask of a shortest path x -> target (target == n: boundary) in the CURRENT
    // integer weights; with toggle, flips the parity of every edge on it.
    //  * A direct edge x-y is a shortest path if w <= (lightest other edge at x) + (lightest
    //    other edge at y): any longer path starts with one such edge and ends with another.
    //    For the boundary the second term is the lightest boundary edge.
    //  * Otherwise Dijkstra with the boundary as a sink.
    uint64_t walk_path(int x, int target, bool toggle) {
        const int want = (target == n) ? -1 : target;
        int kd = -1;                                 // one scan of x: the edge to the target,
        double mx = std::numeric_limits<double>::infinity();   // and x's lightest other edge
        for (int k = adj_off[x], e = adj_off[x + 1]; k < e; ++k) {
            const int t = adj_to[k];
            if (t == want) kd = k;
            else if (t != -1) mx = std::min(mx, adj_w[k]);
        }
        if (kd >= 0) {
            double bound = mx + (target == n ? cur_minB : min_other_fast(target, x));
            if (adj_w[kd] <= bound) {
                if (toggle) flip(adj_eid[kd]);
                return adj_obs[kd];
            }
        }
        if (po_dist.empty()) {
            po_dist.assign(n + 1, std::numeric_limits<double>::infinity());
            po_obs.assign(n + 1, 0);
            po_fin.assign(n + 1, 0);
        }
        if (po_from.empty()) po_from.assign(n + 1, -1);
        auto cmp = std::greater<std::pair<double, int>>();
        po_heap.clear();
        po_dist[x] = 0.0; po_from[x] = -1; po_touched.push_back(x);
        po_heap.push_back({0.0, x});
        bool found = false;
        while (!po_heap.empty()) {
            std::pop_heap(po_heap.begin(), po_heap.end(), cmp);
            auto [d, v] = po_heap.back(); po_heap.pop_back();
            if (po_fin[v]) continue;
            po_fin[v] = 1;
            if (v == target) { found = true; break; }
            if (v == n) continue;
            for (int k = adj_off[v], e = adj_off[v + 1]; k < e; ++k) {
                int u = adj_to[k] == -1 ? n : adj_to[k];
                double nd = d + adj_w[k];
                if (!po_fin[u] && nd < po_dist[u]) {
                    if (po_dist[u] == std::numeric_limits<double>::infinity()) po_touched.push_back(u);
                    po_dist[u] = nd; po_from[u] = k;
                    po_heap.push_back({nd, u});
                    std::push_heap(po_heap.begin(), po_heap.end(), cmp);
                }
            }
        }
        uint64_t mask = 0;
        if (found) {
            for (int v = target; v != x; ) {        // walk back along parent edges
                int k = po_from[v];
                mask ^= adj_obs[k];
                if (toggle) flip(adj_eid[k]);
                v = adj_src[k];
            }
        }
        for (int v : po_touched) { po_dist[v] = std::numeric_limits<double>::infinity(); po_fin[v] = 0; }
        po_touched.clear();
        if (!found) throw std::runtime_error("no path");
        return mask;
    }

    // Is pass 1's matching still optimal after reweighting? At the end of pass 1 the regions are
    // an optimal dual solution. Only reweighted edges changed, and they only got lighter, so it
    // is enough that each of them still respects the regions (then the old duals stay feasible
    // and the old matching stays tight, hence optimal; pass 2 could only pick an equal-weight
    // alternative). For a reweighted edge a-b with new weight w (local = 0 if uncovered or the
    // boundary):
    //   * a, b in the same simple (non-blossom) region: |local(a) - local(b)| <= w, i.e. the
    //     lighter edge is no shortcut inside the region;
    //   * otherwise: local(a) + local(b) <= w, i.e. the regions still don't overlap across it
    //     (conservative inside a blossom, where both sides include the blossom's radius).
    // Must be called before reset(), while the pass-1 regions still exist.
    bool pass1_still_optimal() const {
        for (const auto& sv : saved) {
            const int e = sv.first;
            const int a = adj_src[edge_pos0[e]];
            const int b = edge_pos1[e] >= 0 ? adj_src[edge_pos1[e]] : -1;
            const double w = edge_weight(e);
            const double la = top[a] >= 0 ? local(a) : 0.0;
            const double lb = (b >= 0 && top[b] >= 0) ? local(b) : 0.0;
            const bool same_simple = b >= 0 && top[a] >= 0 && top[a] == top[b] && !R[top[a]].blossom;
            if (same_simple ? std::fabs(la - lb) > w + TOL : la + lb > w + TOL) return false;
        }
        return true;
    }

    void undo_reweights() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
            set_edge_weight(it->first, it->second);
            mark_dirty(it->first, -1);
        }
        saved.clear();
        cur_minB = base_minB;
    }

    void clear_flips() {
        for (int e : flipped) { eflip[e] = 0; eseen[e] = 0; }
        flipped.clear();
    }

    // Lower the weights implied by the edges used an odd number of times, then clear the parities.
    void apply_reweights() {
        for (int e : flipped) {
            if (!eflip[e]) continue;
            for (int r = rule_off[e]; r < rule_off[e + 1]; ++r) {
                int f = rule_f[r];
                double w = rule_w[r];
                double cur = edge_weight(f);
                if (w < cur) {
                    saved.push_back({f, cur});
                    set_edge_weight(f, w);
                    mark_dirty(f, +1);
                    if (edge_pos1[f] < 0) cur_minB = std::min(cur_minB, w);
                }
            }
        }
        clear_flips();
    }

    uint64_t decode_correlated(const std::vector<int>& defects) {
        if (defects.empty()) return 0;
        if ((int)eflip.size() < m) { eflip.assign(m, 0); eseen.assign(m, 0); }
        if (defects.size() == 1) {
            // One lit detector: in both passes the only matching is "to the boundary", so each
            // pass is just a shortest boundary path (pass 2 in the reweighted graph).
            try {
                const int x = defects[0];
                uint64_t obs1 = walk_path(x, n, true);
                apply_reweights();
                if (saved.empty()) return obs1;
                uint64_t obs2 = walk_path(x, n, false);
                undo_reweights();
                return obs2;
            } catch (...) {
                clear_flips();
                undo_reweights();
                throw;
            }
        }
        try {
            // pass 1
            match(defects);
            uint64_t obs1 = 0;
            for (const Edge& p : pairs) obs1 ^= walk_path(p.first, p.second == -1 ? n : p.second, true);
            apply_reweights();                       // (pass-1 regions still alive)
            const bool skip = saved.empty() || pass1_still_optimal();
            reset(defects);
            if (skip) {                              // pass 2 would return pass 1's matching
                undo_reweights();
                return obs1;
            }
            // pass 2 on the reweighted graph
            match(defects);
            uint64_t obs2 = 0;
            for (const Edge& p : pairs) obs2 ^= walk_path(p.first, p.second == -1 ? n : p.second, false);
            reset(defects);
            undo_reweights();
            return obs2;
        } catch (...) {
            reset(defects);
            clear_flips();
            undo_reweights();
            throw;
        }
    }

    uint64_t decode(const std::vector<int>& defects) {
        if (correlated) return decode_correlated(defects);
        if (defects.empty()) return 0;
        uint64_t quick;
        if (shortcut(defects, quick)) return quick;
        try {
            match(defects);
        } catch (...) {
            reset(defects);
            throw;
        }
        uint64_t mask = 0;
        for (const Edge& p : pairs) mask ^= pair_obs(p.first, p.second == -1 ? n : p.second);
        reset(defects);
        return mask;
    }
};

std::string last_error;

}  // namespace

// ---------------------------------------------------------------------- C interface
// Edges: (eu[i], ev[i]) with ev[i] = -1 for the boundary; wint = integer matching weight,
// wf = float weight (log-likelihood), eobs = observable mask.
EXPORT void* sb_create(int n, int m, const int* eu, const int* ev, const double* wint,
                       const double* wf, const uint64_t* eobs) {
    try {
        auto* D = new Decoder();
        D->n = n;
        std::vector<std::vector<std::pair<int, double>>> nbr(n);
        std::vector<std::vector<int>> nbr_edge(n);    // input edge index of each nbr entry
        D->fnbr.assign(n + 1, {});
        D->fw.assign(n + 1, {});
        D->fobs.assign(n + 1, {});
        double wmin = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            int u = eu[i], v = ev[i];
            if (v == -1) {
                nbr[u].push_back({-1, wint[i]}); nbr_edge[u].push_back(i);
                v = n;
            } else {
                nbr[u].push_back({v, wint[i]}); nbr_edge[u].push_back(i);
                nbr[v].push_back({u, wint[i]}); nbr_edge[v].push_back(i);
            }
            D->fnbr[u].push_back(v); D->fw[u].push_back(wf[i]); D->fobs[u].push_back(eobs[i]);
            D->fnbr[v].push_back(u); D->fw[v].push_back(wf[i]); D->fobs[v].push_back(eobs[i]);
            wmin = std::min(wmin, wf[i]);
        }
        D->direct_limit = 2.0 * wmin;
        D->adj_off.assign(n + 1, 0);
        D->m = m;
        D->edge_pos0.assign(m, -1);
        D->edge_pos1.assign(m, -1);
        double wimin = std::numeric_limits<double>::infinity();
        for (int v = 0; v < n; ++v) {
            D->adj_off[v + 1] = D->adj_off[v] + (int)nbr[v].size();
            for (size_t j = 0; j < nbr[v].size(); ++j) {
                auto [u, w] = nbr[v][j];
                int i = nbr_edge[v][j];
                int pos = (int)D->adj_to.size();
                if (D->edge_pos0[i] < 0) D->edge_pos0[i] = pos; else D->edge_pos1[i] = pos;
                if (u == -1) D->base_minB = std::min(D->base_minB, w);
                D->adj_to.push_back(u); D->adj_w.push_back(w); wimin = std::min(wimin, w);
                D->adj_obs.push_back(eobs[i]);
                D->adj_direct.push_back(wf[i] <= D->direct_limit);
                D->adj_eid.push_back(i);
            }
        }
        D->int_direct_limit = 2.0 * wimin;
        // Boundary distances (integer weights) and boundary-path observables (float weights):
        // one multi-source Dijkstra each, started from every boundary edge. Paths never pass
        // through the boundary, matching the decoder's convention.
        using QE = std::pair<double, int>;
        auto boundary_search = [&](bool use_int, std::vector<double>& dist, std::vector<uint64_t>* obs) {
            dist.assign(n, std::numeric_limits<double>::infinity());
            if (obs) obs->assign(n, 0);
            std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
            for (int v = 0; v < n; ++v)
                for (size_t k = 0; k < D->fnbr[v].size(); ++k)
                    if (D->fnbr[v][k] == n) {
                        double w = D->fw[v][k];
                        if (use_int)
                            for (const auto& [u2, w2] : nbr[v]) if (u2 == -1) w = w2;
                        if (w < dist[v]) { dist[v] = w; if (obs) (*obs)[v] = D->fobs[v][k]; pq.push({w, v}); }
                    }
            while (!pq.empty()) {
                auto [d, v] = pq.top(); pq.pop();
                if (d > dist[v]) continue;
                for (size_t k = 0; k < D->fnbr[v].size(); ++k) {
                    int u = D->fnbr[v][k];
                    if (u == n) continue;
                    double w = D->fw[v][k];
                    if (use_int)
                        for (const auto& [u2, w2] : nbr[v]) if (u2 == u) w = w2;
                    if (d + w < dist[u]) {
                        dist[u] = d + w;
                        if (obs) (*obs)[u] = (*obs)[v] ^ D->fobs[v][k];
                        pq.push({dist[u], u});
                    }
                }
            }
        };
        std::vector<double> fdist;
        boundary_search(true, D->bdist, nullptr);
        boundary_search(false, fdist, &D->bobs);
        D->node_ver.assign(n, 0);
        D->top.assign(n, -1);
        D->srcv.assign(n, -1);
        D->trivial_of.assign(n, -1);
        D->base.assign(n, 0.0);
        return D;
    } catch (const std::exception& e) {
        last_error = e.what();
        return nullptr;
    }
}

EXPORT void sb_destroy(void* h) { delete static_cast<Decoder*>(h); }

// data: shots x bytes_per_shot bit-packed detection events (little-endian bit order, as stim/sinter).
// out:  one observable mask per shot. Returns 0 on success, -1 on error (see sb_last_error).
static void decode_range(Decoder* D, const uint8_t* data, int s0, int s1, int bytes_per_shot, uint64_t* out) {
    std::vector<int> defects;
    for (int s = s0; s < s1; ++s) {
        defects.clear();
        const uint8_t* row = data + (size_t)s * bytes_per_shot;
        for (int b = 0; b < bytes_per_shot; ++b) {
            uint8_t byte = row[b];
            while (byte) {
                int bit = __builtin_ctz(byte);
                int det = b * 8 + bit;
                if (det < D->n) defects.push_back(det);
                byte &= byte - 1;
            }
        }
        out[s] = D->decode(defects);
    }
}

EXPORT int sb_decode_batch(void* h, const uint8_t* data, int shots, int bytes_per_shot, uint64_t* out) {
    try {
        decode_range(static_cast<Decoder*>(h), data, 0, shots, bytes_per_shot, out);
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

// Same as sb_decode_batch, split over `threads` threads. Shots are independent and the graph
// is read-only, so each thread decodes a contiguous slice with its own copy of the decoder.
EXPORT int sb_decode_batch_mt(void* h, const uint8_t* data, int shots, int bytes_per_shot,
                              uint64_t* out, int threads) {
    auto* D = static_cast<Decoder*>(h);
    if (threads <= 1 || shots < 2 * threads) return sb_decode_batch(h, data, shots, bytes_per_shot, out);
    try {
        if (!D->workers) D->workers = new std::vector<Decoder*>();
        while ((int)D->workers->size() < threads - 1) D->workers->push_back(D->clone());
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
    std::vector<std::string> errors(threads);
    std::vector<std::thread> pool;
    auto run = [&](int t, Decoder* dec) {
        int s0 = (int)((long long)shots * t / threads), s1 = (int)((long long)shots * (t + 1) / threads);
        try { decode_range(dec, data, s0, s1, bytes_per_shot, out); }
        catch (const std::exception& e) { errors[t] = e.what(); }
    };
    for (int t = 1; t < threads; ++t) pool.emplace_back(run, t, (*D->workers)[t - 1]);
    run(0, D);
    for (auto& th : pool) th.join();
    for (const auto& e : errors)
        if (!e.empty()) { last_error = e; return -1; }
    return 0;
}

EXPORT const char* sb_last_error() { return last_error.c_str(); }

// Switch a decoder to correlated matching. Rule r says: if edge rule_e[r] is used by the first
// matching, edge rule_f[r] may be lowered to integer weight rule_w[r] (indices are input edge
// indices as passed to sb_create). Returns 0 on success.
EXPORT int sb_set_correlations(void* h, int nrules, const int* rule_e, const int* rule_f, const double* rule_w) {
    auto* D = static_cast<Decoder*>(h);
    try {
        std::vector<int> count(D->m + 1, 0);
        for (int r = 0; r < nrules; ++r) {
            if (rule_e[r] < 0 || rule_e[r] >= D->m || rule_f[r] < 0 || rule_f[r] >= D->m)
                throw std::runtime_error("correlation rule refers to an unknown edge");
            ++count[rule_e[r] + 1];
        }
        D->rule_off.assign(D->m + 1, 0);
        for (int e = 0; e < D->m; ++e) D->rule_off[e + 1] = D->rule_off[e] + count[e + 1];
        D->rule_f.assign(nrules, 0);
        D->rule_w.assign(nrules, 0.0);
        std::vector<int> fill(D->rule_off.begin(), D->rule_off.end() - 1);
        for (int r = 0; r < nrules; ++r) {
            int slot = fill[rule_e[r]]++;
            D->rule_f[slot] = rule_f[r];
            D->rule_w[slot] = rule_w[r];
        }
        D->adj_src.assign(D->adj_to.size(), 0);
        for (int v = 0; v < D->n; ++v)
            for (int k = D->adj_off[v]; k < D->adj_off[v + 1]; ++k) D->adj_src[k] = v;
        const double INF = std::numeric_limits<double>::infinity();
        D->top1w.assign(D->n, INF); D->top2w.assign(D->n, INF); D->top1nbr.assign(D->n, -2);
        D->ndirty.assign(D->n, 0);
        for (int v = 0; v < D->n; ++v)
            for (int k = D->adj_off[v]; k < D->adj_off[v + 1]; ++k) {
                if (D->adj_to[k] == -1) continue;
                double w = D->adj_w[k];
                if (w < D->top1w[v]) { D->top2w[v] = D->top1w[v]; D->top1w[v] = w; D->top1nbr[v] = D->adj_to[k]; }
                else if (w < D->top2w[v]) D->top2w[v] = w;
            }
        D->cur_minB = D->base_minB;
        D->correlated = true;
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

// Matched (detector, detector or -1) pairs of the most recently decoded shot, for testing.
EXPORT int sb_last_pairs(void* h, int* out, int max_pairs) {
    auto* D = static_cast<Decoder*>(h);
    int k = 0;
    for (const auto& p : D->pairs) {
        if (k >= max_pairs) break;
        out[2 * k] = p.first; out[2 * k + 1] = p.second; ++k;
    }
    return k;
}
