#!/usr/bin/env python3
"""
Simple test script to verify the preprocessing pipeline works correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def create_test_pgn(filepath, is_lichess=False):
    """Create a simple test PGN file."""
    if is_lichess:
        content = '''[Event "Rated Blitz game"]
[Site "https://lichess.org/test1"]
[Date "2024.01.01"]
[Round "-"]
[White "TestPlayer1"]
[Black "TestPlayer2"]
[Result "1-0"]
[WhiteElo "1850"]
[BlackElo "1900"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7 9. h3 h6 10. Nbd2 Re8 1-0

[Event "Rated Classical game"]
[Site "https://lichess.org/test2"]
[Date "2024.01.01"]
[Round "-"]
[White "TestPlayer3"]
[Black "TestPlayer4"]
[Result "0-1"]
[WhiteElo "1950"]
[BlackElo "1850"]
[Termination "Normal"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Nd3 f5 0-1
'''
    else:
        content = '''[Event "Test LC0 Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "TestPlayer1"]
[Black "TestPlayer2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Test LC0 Game 2"]
[Site "Test"]
[Date "2024.01.01"]
[Round "2"]
[White "TestPlayer3"]
[Black "TestPlayer4"]
[Result "1/2-1/2"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 5. d3 d6 6. e4 e5 7. Nc3 Nc6 8. h3 h6 1/2-1/2
'''
    
    with open(filepath, 'w') as f:
        f.write(content)

def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("Creating test data...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        raw_dir = temp_path / "raw"
        out_dir = temp_path / "output"
        
        # Create directory structure
        (raw_dir / "lc0").mkdir(parents=True)
        (raw_dir / "lichess").mkdir(parents=True)
        
        # Create test PGN files
        create_test_pgn(raw_dir / "lc0" / "test_lc0.pgn", is_lichess=False)
        create_test_pgn(raw_dir / "lichess" / "test_lichess.pgn", is_lichess=True)
        
        print(f"Test data created in: {raw_dir}")
        print(f"LC0 files: {list((raw_dir / 'lc0').glob('*.pgn'))}")
        print(f"Lichess files: {list((raw_dir / 'lichess').glob('*.pgn'))}")
        
        # Test the preprocessing
        print("\nTesting preprocessing...")
        
        # Import and run preprocessing
        try:
            import subprocess
            cmd = [
                sys.executable, "preprocess_tensors.py",
                "--raw-dir", str(raw_dir),
                "--out-dir", str(out_dir),
                "--chunk-size", "100",  # Small chunk size for testing
                "--jobs", "1",  # Single worker for simplicity
                "--mix-passes", "1",
                "--log-level", "INFO",
                "--worker-buffer-multiplier", "2"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("✓ Preprocessing completed successfully!")
                
                # Check output files
                lichess_dir = out_dir / "lichess"
                lc0_dir = out_dir / "lc0"
                
                if lichess_dir.exists():
                    lichess_files = list(lichess_dir.glob("*.npz"))
                    print(f"✓ Lichess output files: {len(lichess_files)} files")
                    for f in lichess_files:
                        print(f"  - {f.name} ({f.stat().st_size} bytes)")
                
                if lc0_dir.exists():
                    lc0_files = list(lc0_dir.glob("*.npz"))
                    print(f"✓ LC0 output files: {len(lc0_files)} files")
                    for f in lc0_files:
                        print(f"  - {f.name} ({f.stat().st_size} bytes)")
                
                # Check metadata
                metadata_file = out_dir / "metadata.json"
                if metadata_file.exists():
                    print(f"✓ Metadata file created: {metadata_file}")
                    with open(metadata_file) as f:
                        import json
                        metadata = json.load(f)
                        print(f"  Metadata: {metadata}")
                
                return True
            else:
                print(f"✗ Preprocessing failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"✗ Error running preprocessing: {e}")
            return False

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1) 