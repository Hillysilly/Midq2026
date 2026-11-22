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

struct Region {
    double y0 = 0.0, t0 = 0.0;            // radius at time t0
    int slope = 0;                        // +1 growing, -1 shrinking, 0 frozen
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

struct Event {
    double t;
    uint64_t seq;
    int kind;     // 0: edge (a, u) may become covered;  1: region a shrinks into something
    int a, u;
    double w;
};
struct EventLater {
    bool operator()(const Event& x, const Event& y) const {
        if (x.t != y.t) return x.t > y.t;
        return x.seq > y.seq;
    }
};

struct Decoder {
    int n = 0;                                            // detectors; node n = boundary (obs graph only)
    std::vector<std::vector<std::pair<int, double>>> nbr; // integer weights; neighbour -1 = boundary
    std::vector<std::vector<int>> fnbr;                   // float graph for observable paths (n+1 nodes)
    std::vector<std::vector<double>> fw;
    std::vector<std::vector<uint64_t>> fobs;
    double direct_limit = 0.0;

    // per-shot state
    std::vector<int> top, srcv, trivial_of;
    std::vector<double> base;
    std::vector<int> touched;
    std::vector<Region> R;
    std::priority_queue<Event, std::vector<Event>, EventLater> heap;
    uint64_t seq = 0;
    double now = 0.0;
    long long events = 0;
    std::vector<Edge> pairs;

    // ------------------------------------------------------------------ helpers
    double y(int r) const { return R[r].y0 + R[r].slope * (now - R[r].t0); }
    void setslope(int r, int s) { R[r].y0 = y(r); R[r].t0 = now; R[r].slope = s; }
    double local(int v) const { return base[v] + y(top[v]); }
    void cover(int v, int r) { top[v] = r; touched.push_back(v); }

    std::vector<int> nodes_of(int r) const {
        std::vector<int> out, stack{r};
        while (!stack.empty()) {
            int X = stack.back(); stack.pop_back();
            if (!R[X].blossom) out.push_back(R[X].src);
            else stack.insert(stack.end(), R[X].children.begin(), R[X].children.end());
            out.insert(out.end(), R[X].shell.begin(), R[X].shell.end());
        }
        return out;
    }

    void push(double t, int kind, int a, int u = -1, double w = 0.0) {
        heap.push(Event{t, ++seq, kind, a, u, w});
    }

    // When will edge a-b (b = -1: boundary) be fully covered?
    bool edge_time(int a, int b, double w, double& t) const {
        int Ta = top[a];
        if (b == -1) {
            if (Ta < 0 || R[Ta].slope <= 0) return false;
            t = now + (w - local(a)) / R[Ta].slope;
            return true;
        }
        int Tb = top[b];
        if (Ta < 0 && Tb < 0) return false;
        if (Ta < 0) { std::swap(a, b); std::swap(Ta, Tb); }
        if (Tb < 0) {
            if (R[Ta].slope <= 0) return false;
            t = now + (w - local(a)) / R[Ta].slope;
            return true;
        }
        if (Ta == Tb) return false;
        int s = R[Ta].slope + R[Tb].slope;
        if (s <= 0) return false;
        t = now + (w - local(a) - local(b)) / s;
        return true;
    }

    void schedule_node(int v) {
        for (const auto& [u, w] : nbr[v]) {
            double t;
            if (edge_time(v, u, w, t)) push(std::max(t, now), 0, v, u, w);
        }
    }

    bool shrink_time(int r, double& t) const {
        const Region& X = R[r];
        if (!X.alive || X.bparent != NONE || X.slope >= 0) return false;
        t = X.shell.empty() ? now + y(r) : now + local(X.shell.back());
        return true;
    }

    void schedule_region(int r) {
        for (int v : nodes_of(r)) schedule_node(v);
        double t;
        if (shrink_time(r, t)) push(std::max(t, now), 1, r);
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

    void dissolve(int rt) {
        std::vector<int> regs, stack{rt};
        while (!stack.empty()) {
            int X = stack.back(); stack.pop_back();
            stack.insert(stack.end(), R[X].tchildren.begin(), R[X].tchildren.end());
            R[X].tparent = NONE; R[X].tedge = {-1, -1}; R[X].tchildren.clear();
            setslope(X, 0);
            regs.push_back(X);
        }
        for (int X : regs) schedule_region(X);
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

        int Bl = (int)R.size();
        R.emplace_back();                                          // (no references held across this)
        R[Bl].blossom = true;
        R[Bl].children = cyc;
        R[Bl].cycle = edges;
        R[Bl].slope = 1; R[Bl].y0 = 0.0; R[Bl].t0 = now;
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
            double yx = R[X].y0;
            for (int v : nodes_of(X)) { base[v] += yx; top[v] = Bl; }
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
        schedule_region(B);
        schedule_region(partner);
    }

    void collide(int A, int B, Edge e) {
        if (R[A].slope < R[B].slope) { std::swap(A, B); e = {e.second, e.first}; }
        if (R[B].slope == 1) {
            int ra = root(A), rb = root(B);
            if (ra == rb) {
                form_blossom(A, B, e);
            } else {
                set_match(A, B, e);
                augment(A); augment(B);
                dissolve(ra); dissolve(rb);
            }
        } else if (R[B].slope == 0) {
            if (R[B].match == BOUNDARY) {
                int ra = root(A);
                set_match(A, B, e);
                augment(A);
                dissolve(ra);
                schedule_region(B);
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
            R[X].y0 = yx; R[X].t0 = now;
            for (int v : nodes_of(X)) { base[v] -= yx; top[v] = X; }
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
        for (const auto& [u, w] : nbr[s]) {
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
        for (int x : defects) {
            int r = (int)R.size();
            R.emplace_back();
            R[r].src = x; R[r].slope = 1;
            trivial_of[x] = r;
            cover(x, r); base[x] = 0.0; srcv[x] = x;
        }
        for (int x : defects) schedule_node(x);

        while (!heap.empty()) {
            Event ev = heap.top(); heap.pop();
            if (++events > MAX_EVENTS) throw std::runtime_error("too many events");
            if (ev.kind == 0) {
                double tt;
                if (!edge_time(ev.a, ev.u, ev.w, tt) || std::fabs(tt - ev.t) > TOL) continue;
                now = std::max(now, ev.t);
                int a = ev.a, u = ev.u;
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
                    for (const auto& [u, w] : nbr[v])
                        if (u != -1 && top[u] >= 0) schedule_node(u);
                    double t2;
                    if (shrink_time(r, t2)) push(std::max(t2, now), 1, r);
                } else if (R[r].blossom) {
                    shatter(r);
                } else {
                    trivial_zero(r);
                }
            }
        }

        // expand blossoms into detector pairs
        std::function<void(int, int)> expand = [&](int Rr, int x) {
            while (R[Rr].blossom) {
                const std::vector<int> ch = R[Rr].children;
                const std::vector<Edge> cyc = R[Rr].cycle;
                int m = (int)ch.size();
                int X = trivial_of[x];
                while (R[X].bparent != Rr) X = R[X].bparent;
                int j = int(std::find(ch.begin(), ch.end(), X) - ch.begin());
                for (int k = 1; k < m; k += 2) {
                    int i1 = (j + k) % m, i2 = (j + k + 1) % m;
                    pairs.push_back(cyc[i1]);
                    expand(ch[i1], cyc[i1].first);
                    expand(ch[i2], cyc[i1].second);
                }
                Rr = X;
            }
        };
        std::vector<int> tops;
        for (int x : defects) {
            int Rr = trivial_of[x];
            while (R[Rr].bparent != NONE) Rr = R[Rr].bparent;
            if (std::find(tops.begin(), tops.end(), Rr) == tops.end()) tops.push_back(Rr);
        }
        std::vector<char> done(R.size(), 0);
        for (int T : tops) {
            if (done[T]) continue;
            if (R[T].match == NONE) throw std::runtime_error("unmatched region at the end");
            Edge me = R[T].medge;
            pairs.push_back(me);
            expand(T, me.first);
            done[T] = 1;
            if (R[T].match != BOUNDARY) { expand(R[T].match, me.second); done[R[T].match] = 1; }
        }
    }

    void reset(const std::vector<int>& defects) {
        for (int v : touched) { top[v] = -1; base[v] = 0.0; srcv[v] = -1; }
        touched.clear();
        for (int x : defects) trivial_of[x] = -1;
        R.clear();
        while (!heap.empty()) heap.pop();
    }

    // ------------------------------------------------------------------ observables
    uint64_t pair_obs(int x, int target) const {
        for (size_t k = 0; k < fnbr[x].size(); ++k)
            if (fnbr[x][k] == target && fw[x][k] <= direct_limit) return fobs[x][k];
        // Dijkstra from x to target; the boundary (node n) is a sink, never passed through.
        std::vector<double> dist(n + 1, std::numeric_limits<double>::infinity());
        std::vector<uint64_t> ob(n + 1, 0);
        std::vector<char> fin(n + 1, 0);
        using QE = std::pair<double, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        dist[x] = 0.0; pq.push({0.0, x});
        while (!pq.empty()) {
            auto [d, v] = pq.top(); pq.pop();
            if (fin[v]) continue;
            fin[v] = 1;
            if (v == target) return ob[v];
            if (v == n) continue;
            for (size_t k = 0; k < fnbr[v].size(); ++k) {
                int u = fnbr[v][k];
                double nd = d + fw[v][k];
                if (!fin[u] && nd < dist[u]) { dist[u] = nd; ob[u] = ob[v] ^ fobs[v][k]; pq.push({nd, u}); }
            }
        }
        throw std::runtime_error("no path");
    }

    uint64_t decode(const std::vector<int>& defects) {
        if (defects.empty()) return 0;
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
        D->nbr.assign(n, {});
        D->fnbr.assign(n + 1, {});
        D->fw.assign(n + 1, {});
        D->fobs.assign(n + 1, {});
        double wmin = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            int u = eu[i], v = ev[i];
            if (v == -1) {
                D->nbr[u].push_back({-1, wint[i]});
                v = n;
            } else {
                D->nbr[u].push_back({v, wint[i]});
                D->nbr[v].push_back({u, wint[i]});
            }
            D->fnbr[u].push_back(v); D->fw[u].push_back(wf[i]); D->fobs[u].push_back(eobs[i]);
            D->fnbr[v].push_back(u); D->fw[v].push_back(wf[i]); D->fobs[v].push_back(eobs[i]);
            wmin = std::min(wmin, wf[i]);
        }
        D->direct_limit = 2.0 * wmin;
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
EXPORT int sb_decode_batch(void* h, const uint8_t* data, int shots, int bytes_per_shot, uint64_t* out) {
    auto* D = static_cast<Decoder*>(h);
    std::vector<int> defects;
    try {
        for (int s = 0; s < shots; ++s) {
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
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

EXPORT const char* sb_last_error() { return last_error.c_str(); }

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
