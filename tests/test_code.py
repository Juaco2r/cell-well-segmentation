from pathlib import Path
from cellwell.pipeline import process_single_file

def test_pipeline_runs_on_example(tmp_path):
    example_tif = Path("data/example/D1_CropMini.tif")
    assert example_tif.exists()

    process_single_file(example_tif)
