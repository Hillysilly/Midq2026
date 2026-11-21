"""Surface-code QEC toolkit for the midq2026 hackathon.

Quick tour:
    from qec import SurfaceCode, NoiseModel, detector_error_model, make_decoder

    code = SurfaceCode(5, variant="css")               # or "xzzx", or SurfaceCode(dx=3, dz=7)
    circuit = code.memory_circuit(NoiseModel.circuit(1e-3), rounds=5, basis="Z")
    dem = detector_error_model(circuit)                # the detector (hyper)graph
    dets, obs = circuit.compile_detector_sampler().sample(10_000, separate_observables=True)
    pred = make_decoder("pymatching", dem).decode_batch(dets)
    print("logical error rate:", (pred != obs).any(axis=1).mean())
"""
from .circuits import detector_error_model, memory_circuit
from .decoders import DECODER_NAMES, make_decoder, sinter_decoders
from .noise import NoiseModel
from .surface_code import SurfaceCode
from .union_find import UnionFindDecoder, UnionFindSinterDecoder

__all__ = [
    "SurfaceCode", "NoiseModel", "memory_circuit", "detector_error_model",
    "make_decoder", "sinter_decoders", "DECODER_NAMES", "UnionFindDecoder", "UnionFindSinterDecoder",
]
