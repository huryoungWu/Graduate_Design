@echo off
setlocal
cd /d "%~dp0"
set PY="D:\Python\python.exe"

echo ============================================================
echo Stage 1: train + save artifacts
echo ============================================================

%PY% -m models.forecasting.itransformer --stage train --epochs 50
%PY% -m models.forecasting.transformer --stage train --epochs 50
%PY% -m models.forecasting.lstm --stage train --epochs 50
%PY% -m models.forecasting.micn --stage train --epochs 50
%PY% -m models.forecasting.moderntcn --stage train --epochs 50
%PY% -m models.forecasting.tcn --stage train --epochs 50
%PY% -m models.forecasting.timesnet --stage train --epochs 50
%PY% -m models.forecasting.seq2seq_lstm --stage train --epochs 50

%PY% -m models.reconstruction.ae --stage train --epochs 50
%PY% -m models.reconstruction.usad --stage train
%PY% -m models.reconstruction.lstm_vae --stage train
%PY% -m models.classical.isolation_forest --protocol framework --stage train
%PY% -m models.classical.dagmm --protocol framework --stage train
%PY% -m models.classical.adt --protocol paper --csv 08_SWaT/SWaT_all_1s.csv
%PY% -m models.reconstruction.anomaly_transformer --stage train --epochs 50
%PY% -m models.reconstruction.dcdetector --stage train --epochs 50
%PY% -m models.reconstruction.sbt --stage train --epochs 50
%PY% -m models.reconstruction.vae --stage train --epochs 50

echo ============================================================
echo Stage 2: anomaly detection from artifacts
echo ============================================================

%PY% -m models.detect --artifact-dir runs/itransformer_swat
%PY% -m models.detect --artifact-dir runs/transformer_swat
%PY% -m models.detect --artifact-dir runs/lstm_swat
%PY% -m models.detect --artifact-dir runs/micn_swat
%PY% -m models.detect --artifact-dir runs/moderntcn_swat
%PY% -m models.detect --artifact-dir runs/tcn_swat
%PY% -m models.detect --artifact-dir runs/timesnet_swat
%PY% -m models.detect --artifact-dir runs/seq2seq_lstm_swat

%PY% -m models.detect --artifact-dir runs/ae_swat
%PY% -m models.detect --artifact-dir runs/usad_swat
%PY% -m models.detect --artifact-dir runs/lstmvae_swat
%PY% -m models.detect --artifact-dir runs/iforest_swat
%PY% -m models.detect --artifact-dir runs/dagmm_swat
%PY% -m models.detect --artifact-dir runs/anomalytransformer_swat
%PY% -m models.detect --artifact-dir runs/dcdetector_swat
%PY% -m models.detect --artifact-dir runs/sbt_swat
%PY% -m models.detect --artifact-dir runs/vae_swat

echo Done.
endlocal
