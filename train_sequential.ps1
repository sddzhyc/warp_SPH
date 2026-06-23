# Sequential training script: trains controller with T=3, T=5, T=7
# Each run uses the best model from the previous run as initialization.
# LR is halved after each run (starting from 0.0001).

$ErrorActionPreference = "Stop"

# ---- Fixed arguments ----
$SceneFile     = "./data/scenes/fish_on_water_new.json"
$Method        = 1
$Iters         = 1
$ControlInterval = 50
$BackwardInterval = 15
$EncoderPath   = "fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth"

# ---- Adjustable parameters ----
$InitialLR  = 0.0001
$TimeValues = @(3, 5, 7)

# ---- State ----
$ModelPath = $null   # no initial model
$LR = $InitialLR

# ---- Temp file to capture output for parsing ----
$TempFile = "train_output_tmp.txt"

foreach ($T in $TimeValues) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Training with T=$T, LR=$LR"       -ForegroundColor Cyan
    Write-Host "  Model path: $ModelPath"            -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Build argument list
    $ArgList = @(
        "./train_controller_diff.py",
        "--scene_file", $SceneFile,
        "--lr", "$LR",
        "--method", "$Method",
        "--iters", "$Iters",
        "--difffr",
        "--norm_grad",
        "--custom_grad",
        "--t", "$T",
        "--control_interval", "$ControlInterval",
        "--bi", "$BackwardInterval",
        "--use_fluid",
        "--encoder_path", $EncoderPath,
        "--random_start",
        "--use_relative_state",
        "--time_in_state",
        "--add_gru",
        "--use_v_loss"
    )

    if ($ModelPath) {
        $ArgList += "--model_path"
        $ArgList += $ModelPath
    }

    # Run training, capturing output to both console and temp file
    & python $ArgList 2>&1 | Tee-Object -FilePath $TempFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }

    # Parse the [BEST_MODEL] line from output
    $OutputContent = Get-Content $TempFile -Raw
    $Match = [regex]::Match($OutputContent, '\[BEST_MODEL\]\s+(.+)')
    if ($Match.Success) {
        $ModelPath = $Match.Groups[1].Value.Trim()
        Write-Host ""
        Write-Host ">>> Best model: $ModelPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Could not find [BEST_MODEL] in training output!" -ForegroundColor Red
        exit 1
    }

    # Halve LR for next run
    $LR = [math]::Round($LR * 0.5, 8)
}

# Clean up temp file
Remove-Item $TempFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All training done!"                    -ForegroundColor Cyan
Write-Host "  Final best model: $ModelPath"          -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
