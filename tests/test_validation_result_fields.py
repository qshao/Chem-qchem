from qchem_gnn.contrastive_pretrain import contrastive_pretrain_on_dataset
from qchem_gnn.encoder3d import Conformer3DEncoder
from qchem_gnn.teacher_heads import QuantumTeacherHeads
from tests._validation_fixtures import make_tiny_quantum_dataset


def test_pretrain_result_exposes_teacher_and_encoder(tmp_path):
    dataset = make_tiny_quantum_dataset(tmp_path)
    result = contrastive_pretrain_on_dataset(
        dataset,
        hidden_dim=16,
        hidden_dim_3d=16,
        epochs=2,
        batch_size=4,
        teacher_weight=1.0,
        conformer_pool_mode="energy",
        seed=0,
    )
    assert isinstance(result.teacher, QuantumTeacherHeads)
    assert isinstance(result.encoder3d, Conformer3DEncoder)
