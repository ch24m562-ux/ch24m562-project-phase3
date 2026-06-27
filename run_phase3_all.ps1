# run_phase3_all.ps1
param(
    [string]$Policy = "rlinv",
    [string]$Sites = "all",
    [string]$Tag = "phase3",
    [string]$Lead = "normal",
    [int]$Timesteps = 400000,
    [int]$MultiTimesteps = 500000,
    [string]$LogBase = "runs",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$seeds = @(42, 123, 777, 7, 13, 21, 99, 314, 500, 999)

if ($Sites -eq "all") {
    $siteList = 1..10 | ForEach-Object { "site$_" }
} else {
    $siteList = $Sites -split ","
}

$failed = @()

foreach ($site in $siteList) {
    foreach ($seed in $seeds) {

        Write-Host "=== $Policy | $site | seed=$seed | lead=$Lead ===" -ForegroundColor Cyan

        if ($Policy -eq "multi") {
            $outDir = "$LogBase/rlinv_multi"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        } elseif ($Policy -eq "trackb") {
            $outDir = "$LogBase/trackb"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        } elseif ($Policy -eq "a5") {
            $outDir = "$LogBase/ablation_a5"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        } elseif ($Policy -eq "a6") {
            $outDir = "$LogBase/ablation_a6"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        } elseif ($Policy -eq "a7") {
            $outDir = "$LogBase/ablation_a7"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        } else {
            $outDir = "$LogBase/rlinv"
            $modelPath = "$outDir/${site}_s${seed}_final.zip"
        }

        if ($Resume -and (Test-Path $modelPath)) {
            Write-Host "SKIP existing: $modelPath" -ForegroundColor Yellow
            continue
        }

        try {
            if ($Policy -eq "rlinv") {
                python src/train/train_rl_inv.py `
                    --site $site --lead $Lead --seed $seed `
                    --timesteps $Timesteps --tag $Tag `
                    --logdir $outDir
            }
            elseif ($Policy -eq "multi") {
                python src/train/train_rl_inv.py `
                    --site $site --lead multi --seed $seed `
                    --timesteps $MultiTimesteps --tag $Tag `
                    --logdir $outDir
            }
            elseif ($Policy -eq "trackb") {
                python src/train/train_track_b.py `
                    --site $site --lead $Lead --seed $seed `
                    --timesteps $Timesteps --tag $Tag `
                    --logdir $outDir
            }
            elseif ($Policy -eq "a5") {
                python src/train/train_ablation_a5.py `
                    --site $site --lead $Lead --seed $seed `
                    --timesteps $Timesteps --tag $Tag `
                    --logdir $outDir
            }
            elseif ($Policy -eq "a6") {
                python src/train/train_ablation_a6.py `
                    --site $site --lead $Lead --seed $seed `
                    --timesteps $Timesteps --tag $Tag `
                    --logdir $outDir
            }
            elseif ($Policy -eq "a7") {
                python src/train/train_ablation_a7.py `
                    --site $site --lead $Lead --seed $seed `
                    --timesteps $Timesteps --tag $Tag `
                    --logdir $outDir
            }
            else {
                throw "Unknown policy: $Policy"
            }

            if ($LASTEXITCODE -ne 0) {
                throw "Python failed with exit code $LASTEXITCODE"
            }
        }
        catch {
            Write-Host "FAILED: $Policy | $site | seed=$seed" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
            $failed += "$Policy,$site,$seed,$Lead"
        }
    }
}

if ($failed.Count -gt 0) {
    $failed | Out-File "results/failed_runs_$Policy.txt"
    Write-Host "Some runs failed. See results/failed_runs_$Policy.txt" -ForegroundColor Red
} else {
    Write-Host "Done: $Policy on $Sites" -ForegroundColor Green
}