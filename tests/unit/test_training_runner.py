from scripts.model.run_training import _parse_log


def test_training_log_parser_extracts_losses_throughput_and_peak_memory() -> None:
    log = """
Iter 0: Val loss 2.500, Val took 1.200s
Iter 10: Train loss 1.750, Learning Rate 1.000e-05, It/sec 0.800, Tokens/sec 125.500, Trained Tokens 1000, Peak mem 4.125 GB
Iter 100: Val loss 0.750, Val took 1.000s
Iter 100: Train loss 0.500, Learning Rate 1.000e-05, It/sec 0.900, Tokens/sec 140.000, Trained Tokens 10000, Peak mem 4.500 GB
"""
    parsed = _parse_log(log)
    assert parsed["final_train_loss"] == 0.5
    assert parsed["final_validation_loss"] == 0.75
    assert parsed["peak_memory_gb"] == 4.5
    assert len(parsed["train_reports"]) == 2
