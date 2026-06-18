<# 
PowerShell version of scripts/run_overnight_training.sh for native Windows execution.
#>

[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RunArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$script:ROOT_DIR = (Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")).Path
Set-Location $script:ROOT_DIR
$script:RUN_LOG_FILE = $null
$script:GLOBAL_LOG_FILE = $null

if (-not ("StreamForwarder" -as [type])) {
  Add-Type -TypeDefinition @"
using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;

public static class StreamForwarder
{
    public static Task ForwardAsync(Stream stream, string stepLog, string runLog, string globalLog)
    {
        return Task.Run(() =>
        {
            var buffer = new byte[4096];
            var encoding = Console.OutputEncoding ?? Encoding.UTF8;
            using (var stepWriter = string.IsNullOrEmpty(stepLog) ? null : new StreamWriter(new FileStream(stepLog, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), encoding))
            using (var runWriter = string.IsNullOrEmpty(runLog) ? null : new StreamWriter(new FileStream(runLog, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), encoding))
            using (var globalWriter = string.IsNullOrEmpty(globalLog) ? null : new StreamWriter(new FileStream(globalLog, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), encoding))
            {
                if (stepWriter != null) stepWriter.AutoFlush = true;
                if (runWriter != null) runWriter.AutoFlush = true;
                if (globalWriter != null) globalWriter.AutoFlush = true;

                int count;
                while ((count = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    var text = encoding.GetString(buffer, 0, count);
                    Console.Write(text);
                    if (stepWriter != null) stepWriter.Write(text);
                    if (runWriter != null) runWriter.Write(text);
                    if (globalWriter != null) globalWriter.Write(text);
                }
            }
        });
    }
}
"@
}

function Write-RunLog {
  param([string]$Message)

  $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
  Write-Host $line
  if ($script:RUN_LOG_FILE) {
    Add-Content -Path $script:RUN_LOG_FILE -Value $line -ErrorAction SilentlyContinue
  }
  if ($script:GLOBAL_LOG_FILE) {
    Add-Content -Path $script:GLOBAL_LOG_FILE -Value $line -ErrorAction SilentlyContinue
  }
}

function Write-StepLog {
  param([string]$Message, [string]$StepLog)
  if ($null -eq $Message) {
    return
  }
  Write-Host $Message
  if ($script:RUN_LOG_FILE) {
    Add-Content -Path $script:RUN_LOG_FILE -Value $Message -ErrorAction SilentlyContinue
  }
  if ($script:GLOBAL_LOG_FILE) {
    Add-Content -Path $script:GLOBAL_LOG_FILE -Value $Message -ErrorAction SilentlyContinue
  }
  if ($StepLog) {
    Add-Content -Path $StepLog -Value $Message -ErrorAction SilentlyContinue
  }
}

function ConvertTo-CommandLineArgument {
  param([string]$Value)

  if ($null -eq $Value) {
    return '""'
  }
  if ($Value -notmatch '[\s"]') {
    return $Value
  }
  return '"' + ($Value -replace '\\(?=\\*")', '$0$0' -replace '"', '\"') + '"'
}

function Parse-Bool {
  param([string]$Name, [int]$DefaultValue = 0)

  $raw = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($raw)) {
    return $DefaultValue
  }

  switch -Regex ($raw.Trim().ToLowerInvariant()) {
    "^(1|true|yes|on)$" { return 1 }
    "^(0|false|no|off)$" { return 0 }
    default { return $DefaultValue }
  }
}

function Normalize-PythonPath {
  param([string]$Candidate)

  if ([string]::IsNullOrWhiteSpace($Candidate)) {
    return $null
  }

  $candidate = $Candidate.Trim().Trim('"')
  if ($candidate -match "^/([a-zA-Z])/(.+)") {
    $drive = $matches[1].ToUpper()
    $rest = ($matches[2] -replace '/', '\')
    return "$drive`:\$rest"
  }
  return $candidate
}

function Test-PythonCandidate {
  param([string]$Candidate, [bool]$RequireGpu)

  if (-not (Test-Path $Candidate -PathType Leaf)) {
    return $false
  }

  if (-not $RequireGpu) {
    return $true
  }

  $null = & $Candidate -c "import torch" 2>$null
  return ($LASTEXITCODE -eq 0)
}

function Resolve-Python {
  $envPython = Normalize-PythonPath -Candidate ([Environment]::GetEnvironmentVariable("PYTHON"))
  $requireGpu = (Parse-Bool -Name "REQUIRE_GPU" -DefaultValue 1) -eq 1

  if (-not [string]::IsNullOrWhiteSpace($envPython)) {
    $candidate = $envPython
    if ($candidate -match "^(python|python3|py)$") {
      $candidate = (Get-Command $candidate -ErrorAction SilentlyContinue).Source
    }
    if (Test-PythonCandidate -Candidate $candidate -RequireGpu:$requireGpu) {
      return (Resolve-Path $candidate).Path
    }

    Write-Host "Configured PYTHON '$envPython' does not satisfy requirements; probing alternatives."
  }

  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("CONDA_PREFIX"))) {
    $condaPrefix = [Environment]::GetEnvironmentVariable("CONDA_PREFIX")
    $candidates += (Join-Path $condaPrefix "python.exe")
    $candidates += (Join-Path $condaPrefix "bin" "python")
    $candidates += (Join-Path $condaPrefix "bin" "python.exe")
    $candidates += (Join-Path $condaPrefix "Scripts" "python")
    $candidates += (Join-Path $condaPrefix "Scripts" "python.exe")
  }

  $user = $env:USERNAME
  $candidates += "C:\Users\${user}\miniconda3\envs\early-icu-mortality\python.exe"
  $candidates += "C:\Users\${user}\miniconda3\envs\early-icu-mortality\python"
  $candidates += "C:\Users\${user}\miniconda3\python.exe"
  $candidates += "C:\Users\${user}\miniconda3\python"
  $candidates += "C:\Users\${user}\anaconda3\envs\early-icu-mortality\python.exe"
  $candidates += "C:\Users\${user}\anaconda3\envs\early-icu-mortality\python"
  $candidates += "C:\Users\${user}\anaconda3\python.exe"
  $candidates += "C:\Users\${user}\anaconda3\python"
  $candidates += "python"
  $candidates += "python3"
  $candidates += "py"

  foreach ($candidate in $candidates) {
    $resolved = $candidate
    if ($candidate -match "^(python|python3|py)$") {
      $resolved = (Get-Command $candidate -ErrorAction SilentlyContinue).Source
    }
    elseif (-not (Test-Path $candidate)) {
      continue
    }

    if (Test-PythonCandidate -Candidate $resolved -RequireGpu:$requireGpu) {
      return (Resolve-Path $resolved).Path
    }
  }

  throw "Unable to locate python. Set PYTHON explicitly. For example: PYTHON=C:\Users\${user}\miniconda3\envs\early-icu-mortality\python.exe"
}

function Get-ChildProcessIds {
  param([int]$ParentId)

  $childIds = @()
  try {
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $ParentId }
    foreach ($child in $children) {
      $childIds += [int]$child.ProcessId
      $childIds += Get-ChildProcessIds -ParentId ([int]$child.ProcessId)
    }
  } catch {
    # Query may fail for transient processes; ignore.
  }
  return $childIds
}

function Stop-ProcessTree {
  param([int]$RootPid)

  $stack = @($RootPid)
  while ($stack.Count -gt 0) {
    $current = $stack[0]
    $stack = $stack[1..($stack.Count - 1)]
    try {
      $children = Get-ChildProcessIds -ParentId $current
      if ($children) {
        $stack += $children
      }
      Stop-Process -Id $current -Force -ErrorAction SilentlyContinue
    } catch {
      # best effort
    }
  }
}

function Stop-ActiveProcesses {
  foreach ($pid in @($script:RUNNING_PROCESS_IDS.ToArray())) {
    Stop-ProcessTree -RootPid $pid
    $null = $script:RUNNING_PROCESS_IDS.Remove($pid)
  }
}

function Invoke-LoggedCommand {
  param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [string[]]$Command
  )

  Write-RunLog "START $Name"
  $stepLog = Join-Path $script:LOG_DIR ("$Name.log")
  if (Test-Path $stepLog) {
    Remove-Item -Force $stepLog
  }

  $outPath = Join-Path $script:LOG_DIR "$Name.log"
  $filePath = $Command[0]
  $argList = @()
  if ($Command.Count -gt 1) {
    $argList = $Command[1..($Command.Count - 1)]
  }
  if (Test-Path $outPath) {
    Remove-Item -Force $outPath
  }

  $argDisplay = if ($argList.Count -gt 0) { $argList -join " " } else { "" }
  Write-RunLog "Running: $filePath $argDisplay"

  $exitCode = 1
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $filePath
  $psi.WorkingDirectory = $script:ROOT_DIR
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.CreateNoWindow = $true
  if ($argList.Count -gt 0) {
    $psi.Arguments = ($argList | ForEach-Object { ConvertTo-CommandLineArgument $_ }) -join " "
  }
  $process = [System.Diagnostics.Process]::new()
  $process.StartInfo = $psi
  try {
    [void]$process.Start()
    [void]$script:RUNNING_PROCESS_IDS.Add($process.Id)

    $stdoutTask = [StreamForwarder]::ForwardAsync($process.StandardOutput.BaseStream, $outPath, $script:RUN_LOG_FILE, $script:GLOBAL_LOG_FILE)
    $stderrTask = [StreamForwarder]::ForwardAsync($process.StandardError.BaseStream, $outPath, $script:RUN_LOG_FILE, $script:GLOBAL_LOG_FILE)

    while (-not $process.WaitForExit(1000)) {
      if ($script:INTERRUPTED) {
        Stop-ProcessTree -RootPid $process.Id
        throw "Interrupted while running step '$Name'."
      }
    }

    [Threading.Tasks.Task]::WaitAll(@($stdoutTask, $stderrTask))
    $exitCode = $process.ExitCode
    [void]$script:RUNNING_PROCESS_IDS.Remove($process.Id)
  } finally {
    if ($null -ne $process) {
      if (-not $process.HasExited) {
        Stop-ProcessTree -RootPid $process.Id
      }
      [void]$script:RUNNING_PROCESS_IDS.Remove($process.Id)
      $process.Dispose()
    }
  }

  if ($exitCode -ne 0) {
    throw "Command failed in step '$Name' with exit code $exitCode."
  }

  Write-RunLog "DONE  $Name"
}

function Invoke-LoggedPythonCode {
  param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [string]$Code,
    [string[]]$Arguments = @()
  )

  $tmp = New-TemporaryFile
  try {
    Set-Content -Path $tmp.FullName -Value $Code -Encoding UTF8 -NoNewline
    $cmd = @($script:PYTHON, $tmp.FullName)
    if ($Arguments.Count -gt 0) {
      $cmd += $Arguments
    }
    Invoke-LoggedCommand -Name $Name -Command $cmd
  } finally {
    if (Test-Path $tmp) {
      Remove-Item -Force $tmp
    }
  }
}

function Invoke-PythonSnippet {
  param(
    [Parameter(Mandatory)] [string]$Code,
    [string[]]$Arguments = @()
  )

  $tmp = New-TemporaryFile
  try {
    Set-Content -Path $tmp.FullName -Value $Code -Encoding UTF8 -NoNewline
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & $script:PYTHON $tmp.FullName @Arguments 2>&1
    return @{ ExitCode = $LASTEXITCODE; Output = ($out -join "`n") }
  } finally {
    if ($null -ne $oldErrorActionPreference) {
      $ErrorActionPreference = $oldErrorActionPreference
    }
    if (Test-Path $tmp) {
      Remove-Item -Force $tmp
    }
  }
}

function Has-Files {
  param([string[]]$Files)

  foreach ($file in $Files) {
    if (-not (Test-Path $file)) {
      return $false
    }
  }
  return $true
}

function Require-File {
  param([string]$Path, [string]$Description)

  if (-not (Test-Path $Path)) {
    Write-RunLog "Missing required ${Description}: ${Path}"
    throw "Missing required file: ${Path}"
  }
}

function Csv-HasColumns {
  param([string]$Path, [string[]]$RequiredColumns)

  if (-not (Test-Path $Path)) {
    return @{ Exists = $false; Missing = "file_not_found" }
  }

  $code = @'
import csv
import sys

path = sys.argv[1]
required = sys.argv[2:]

with open(path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f), [])

missing = [name for name in required if name not in header]
if missing:
    print(",".join(missing))
    raise SystemExit(1)
'@
  $result = Invoke-PythonSnippet -Code $code -Arguments (@($Path) + $RequiredColumns)
  if ($result.ExitCode -ne 0) {
    return @{ Exists = $true; Missing = $result.Output.Trim() }
  }
  return @{ Exists = $true; Missing = "" }
}

function Csv-HasRows {
  param([string]$Path)

  if (-not (Test-Path $Path)) {
    return 0
  }

  $code = @'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    if header is None:
        print(0)
        raise SystemExit(1)
    for i, _ in enumerate(reader, start=1):
        if i >= 1:
            print(i)
            raise SystemExit(0)
    print(0)
    raise SystemExit(1)
'@
  $result = Invoke-PythonSnippet -Code $code -Arguments @($Path)
  if ($result.ExitCode -ne 0) {
    return 0
  }
  $rows = 0
  [void][int]::TryParse(($result.Output -split "`n" | Select-Object -First 1).Trim(), [ref]$rows)
  return $rows
}

function Validate-CsvColumns {
  param([string]$Path, [string]$Description, [string[]]$RequiredColumns)

  $check = Csv-HasColumns -Path $Path -RequiredColumns $RequiredColumns
  if (-not $check.Exists) {
    return @{ Valid = $false; Missing = "file_not_found" }
  }
  if ($check.Missing) {
    Write-RunLog "Invalid cached file: $Description is missing required column(s): $($check.Missing)"
    return @{ Valid = $false; Missing = $check.Missing }
  }
  return @{ Valid = $true; Missing = "" }
}

function Validate-CsvColumnsAndRows {
  param([string]$Path, [string]$Description, [string[]]$RequiredColumns)

  $colCheck = Validate-CsvColumns -Path $Path -Description $Description -RequiredColumns $RequiredColumns
  if (-not $colCheck.Valid) {
    return $false
  }
  $rowCount = Csv-HasRows -Path $Path
  if ($rowCount -le 0) {
    Write-RunLog "Invalid cached file: $Description has no data rows (rows: $rowCount)."
    return $false
  }
  return $true
}

function Ensure-PreprocessedDataset {
  param([string]$Path, [string]$Description, [string[]]$RequiredColumns)

  if ($script:SKIP_PREPROCESS -eq 1) {
    Require-File -Path $Path -Description $Description
    if (-not (Validate-CsvColumnsAndRows -Path $Path -Description $Description -RequiredColumns $RequiredColumns)) {
      return 1
    }
    return 0
  }

  if (-not (Validate-CsvColumnsAndRows -Path $Path -Description $Description -RequiredColumns $RequiredColumns)) {
    Write-RunLog "Regenerating $Description from preprocessing step."
    return 2
  }
  return 0
}

function Run-StepIfMissing {
  param(
    [Parameter(Mandatory)] [string]$Name,
    [string[]]$RequiredFiles,
    [Parameter(Mandatory)] [string[]]$Command
  )

  if (
    $script:SKIP_EXISTING_ARTIFACTS -eq 1 -and
    $null -ne $RequiredFiles -and
    $RequiredFiles.Count -gt 0 -and
    (Has-Files -Files $RequiredFiles)
  ) {
    Write-RunLog "SKIP  $Name (cached artifacts already exist)"
    return
  }

  Invoke-LoggedCommand -Name $Name -Command $Command
}

function Add-PreprocessArgs {
  param([string[]]$Command)

  return @($Command) + @(
    "--imputation-strategy"
    $script:PREPROCESS_IMPUTATION_STRATEGY
    "--mice-max-iter"
    $script:MICE_MAX_ITER
    "--mice-n-estimators"
    $script:MICE_N_ESTIMATORS
  )
}

$script:REQUIRE_GPU = Parse-Bool -Name "REQUIRE_GPU" -DefaultValue 1
$script:SKIP_PREPROCESS = Parse-Bool -Name "SKIP_PREPROCESS" -DefaultValue 0
$script:SKIP_EXISTING_ARTIFACTS = Parse-Bool -Name "SKIP_EXISTING_ARTIFACTS" -DefaultValue 1
$script:RUN_MODEL_SUITE = Parse-Bool -Name "RUN_MODEL_SUITE" -DefaultValue 1
$script:RUN_XGBOOST_ABLATION = Parse-Bool -Name "RUN_XGBOOST_ABLATION" -DefaultValue 1
$script:RUN_LEGACY_XGBOOST_ENSEMBLE = Parse-Bool -Name "RUN_LEGACY_XGBOOST_ENSEMBLE" -DefaultValue 1
$script:RUN_MAFNET = Parse-Bool -Name "RUN_MAFNET" -DefaultValue 1
$script:RUN_MAFNET_ABLATIONS = Parse-Bool -Name "RUN_MAFNET_ABLATIONS" -DefaultValue 1
$script:RUN_FINAL_SUMMARY = Parse-Bool -Name "RUN_FINAL_SUMMARY" -DefaultValue 1
$script:RUN_PYTEST = Parse-Bool -Name "RUN_PYTEST" -DefaultValue 1
$script:PREPROCESS_IMPUTATION_STRATEGY = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("PREPROCESS_IMPUTATION_STRATEGY"))) { "median" } else { [Environment]::GetEnvironmentVariable("PREPROCESS_IMPUTATION_STRATEGY") }
$script:MICE_MAX_ITER = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("MICE_MAX_ITER"))) { "3" } else { [Environment]::GetEnvironmentVariable("MICE_MAX_ITER") }
$script:MICE_N_ESTIMATORS = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("MICE_N_ESTIMATORS"))) { "10" } else { [Environment]::GetEnvironmentVariable("MICE_N_ESTIMATORS") }

$script:PYTHON = Resolve-Python
Write-RunLog "Using python: $script:PYTHON"

$script:ORIGINAL_PYTHONUNBUFFERED = [Environment]::GetEnvironmentVariable("PYTHONUNBUFFERED")
[Environment]::SetEnvironmentVariable("PYTHONUNBUFFERED", "1", "Process")

$script:ORIGINAL_PYTHONWARNINGS = [Environment]::GetEnvironmentVariable("PYTHONWARNINGS")
$requestsWarningFilter = "ignore:Unable to find acceptable character detection dependency"
if ([string]::IsNullOrWhiteSpace($script:ORIGINAL_PYTHONWARNINGS)) {
  [Environment]::SetEnvironmentVariable("PYTHONWARNINGS", $requestsWarningFilter, "Process")
} elseif ($script:ORIGINAL_PYTHONWARNINGS -notlike "*$requestsWarningFilter*") {
  [Environment]::SetEnvironmentVariable("PYTHONWARNINGS", "$script:ORIGINAL_PYTHONWARNINGS,$requestsWarningFilter", "Process")
}

$runName = if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("RUN_NAME"))) {
  [Environment]::GetEnvironmentVariable("RUN_NAME")
} elseif ($null -ne $RunArgs -and $RunArgs.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($RunArgs[0])) {
  $RunArgs[0]
} else {
  "overnight_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
}

$script:RESULTS_ROOT = if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("RESULTS_ROOT"))) {
  [Environment]::GetEnvironmentVariable("RESULTS_ROOT")
} else {
  Join-Path $script:ROOT_DIR ("results\$runName")
}
$script:LOG_DIR = Join-Path $script:RESULTS_ROOT "logs"
$script:CONFIG_DIR = Join-Path $script:RESULTS_ROOT "model_configs_gpu"

$script:TARGET_COL = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("TARGET_COL"))) { "mortality" } else { [Environment]::GetEnvironmentVariable("TARGET_COL") }
$script:SEED = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("SEED"))) { "42" } else { [Environment]::GetEnvironmentVariable("SEED") }
$script:BOOTSTRAP_ITERATIONS = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("BOOTSTRAP_ITERATIONS"))) { "200" } else { [Environment]::GetEnvironmentVariable("BOOTSTRAP_ITERATIONS") }
$script:XGBOOST_ABLATION_ESTIMATORS = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("XGBOOST_ABLATION_ESTIMATORS"))) { "1000" } else { [Environment]::GetEnvironmentVariable("XGBOOST_ABLATION_ESTIMATORS") }
$script:XGBOOST_EARLY_STOPPING_ROUNDS = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("XGBOOST_EARLY_STOPPING_ROUNDS"))) { "50" } else { [Environment]::GetEnvironmentVariable("XGBOOST_EARLY_STOPPING_ROUNDS") }
$script:LEGACY_XGBOOST_ENSEMBLE_SIZE = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("LEGACY_XGBOOST_ENSEMBLE_SIZE"))) { "7" } else { [Environment]::GetEnvironmentVariable("LEGACY_XGBOOST_ENSEMBLE_SIZE") }

$script:BASELINE_FEATURES_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("BASELINE_FEATURES_PATH"))) { "data/processed/extracted_features.csv" } else { [Environment]::GetEnvironmentVariable("BASELINE_FEATURES_PATH") }
$script:EXPANDED_FEATURES_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("EXPANDED_FEATURES_PATH"))) { "data/processed/extracted_features_expanded.csv" } else { [Environment]::GetEnvironmentVariable("EXPANDED_FEATURES_PATH") }
$script:BASELINE_XGBOOST_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("BASELINE_XGBOOST_PATH"))) { "data/processed/preprocessed_xgboost_features.csv" } else { [Environment]::GetEnvironmentVariable("BASELINE_XGBOOST_PATH") }
$script:EXPANDED_XGBOOST_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("EXPANDED_XGBOOST_PATH"))) { "data/processed/preprocessed_xgboost_expanded_features.csv" } else { [Environment]::GetEnvironmentVariable("EXPANDED_XGBOOST_PATH") }
$script:MAFNET_EVENTS_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("MAFNET_EVENTS_PATH"))) { "data/processed/first6h_events.csv" } else { [Environment]::GetEnvironmentVariable("MAFNET_EVENTS_PATH") }
$script:MAFNET_COHORT_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("MAFNET_COHORT_PATH"))) { "data/processed/final_cohort.csv" } else { [Environment]::GetEnvironmentVariable("MAFNET_COHORT_PATH") }
$script:MAFNET_FEATURES_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("MAFNET_FEATURES_PATH"))) { $script:EXPANDED_FEATURES_PATH } else { [Environment]::GetEnvironmentVariable("MAFNET_FEATURES_PATH") }

$script:COHORT_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("COHORT_PATH"))) { "data/processed/final_cohort.csv" } else { [Environment]::GetEnvironmentVariable("COHORT_PATH") }
$script:COHORT_IDS_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("COHORT_IDS_PATH"))) { "data/processed/cohort_stay_ids.csv" } else { [Environment]::GetEnvironmentVariable("COHORT_IDS_PATH") }
$script:BASELINE_FEATURE_STATS_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("BASELINE_FEATURE_STATS_PATH"))) {
  Join-Path (Split-Path $script:BASELINE_FEATURES_PATH) "feature_statistics.csv"
} else {
  [Environment]::GetEnvironmentVariable("BASELINE_FEATURE_STATS_PATH")
}
$script:EXPANDED_FEATURE_STATS_PATH = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("EXPANDED_FEATURE_STATS_PATH"))) {
  Join-Path (Split-Path $script:EXPANDED_FEATURES_PATH) "feature_statistics_expanded.csv"
} else {
  [Environment]::GetEnvironmentVariable("EXPANDED_FEATURE_STATS_PATH")
}

$script:RUN_LOG_FILE = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("RUN_LOG_FILE"))) {
  Join-Path $script:RESULTS_ROOT "log.txt"
} else {
  [Environment]::GetEnvironmentVariable("RUN_LOG_FILE")
}
$script:GLOBAL_LOG_FILE = if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("GLOBAL_LOG_FILE"))) {
  Join-Path $script:ROOT_DIR "log.txt"
} else {
  [Environment]::GetEnvironmentVariable("GLOBAL_LOG_FILE")
}

$script:TMPDIR = Join-Path $script:RESULTS_ROOT "tmp"
$env:TMPDIR = $script:TMPDIR
$env:TMP = $script:TMPDIR
$env:TEMP = $script:TMPDIR
$script:PYTEST_CACHE_DIR = Join-Path $script:RESULTS_ROOT ".pytest_cache"
$script:PYTEST_BASE_TEMP_DIR = Join-Path $script:RESULTS_ROOT "pytest_tmp"

New-Item -ItemType Directory -Force -Path @(
  $script:RESULTS_ROOT,
  $script:LOG_DIR,
  $script:CONFIG_DIR,
  $script:TMPDIR,
  $script:PYTEST_CACHE_DIR,
  $script:PYTEST_BASE_TEMP_DIR
) | Out-Null
foreach ($logFile in @($script:RUN_LOG_FILE, $script:GLOBAL_LOG_FILE)) {
  if (-not (Test-Path $logFile)) {
    New-Item -ItemType File -Path $logFile | Out-Null
  }
}

Write-RunLog "Writing run output to $($script:RUN_LOG_FILE) and $($script:GLOBAL_LOG_FILE)"
Write-RunLog "Run name: $runName"
Write-RunLog "Results root: $($script:RESULTS_ROOT)"

$script:RUNNING_PROCESS_IDS = New-Object 'System.Collections.Generic.HashSet[int]'
$script:INTERRUPTED = $false

$cancelEvent = [ConsoleCancelEventHandler]{
  param($sender, $eventArgs)
  $eventArgs.Cancel = $true
  $script:INTERRUPTED = $true
  Write-RunLog "Received interrupt. Stopping all running jobs..."
  Stop-ActiveProcesses
  exit 130
}
try {
  [Console]::CancelKeyPress += $cancelEvent
} catch {
  Write-RunLog "Warning: Console.CancelKeyPress is not available in this PowerShell host; Ctrl+C handling is limited."
}

if ($script:REQUIRE_GPU -eq 1) {
  Invoke-LoggedPythonCode -Name "check_torch_cuda" -Code @'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is False")
print("CUDA available:", torch.cuda.get_device_name(0))
'@
}

$gpuConfigCode = @'
from pathlib import Path
import shutil
import sys
import yaml

target = Path(sys.argv[1])
source = Path("configs/models")
target.mkdir(parents=True, exist_ok=True)
for path in source.glob("*.yaml"):
    shutil.copy2(path, target / path.name)

def update_yaml(name, updater):
    path = target / name
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    updater(data.setdefault("model", {}).setdefault("params", {}))
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

update_yaml("xgboost.yaml", lambda p: (p.update({"device": "cuda", "tree_method": "hist"})))
update_yaml("catboost.yaml", lambda p: p.update({"task_type": "GPU", "devices": "0"}))

import os
if os.getenv("ENABLE_LIGHTGBM_GPU", "0") == "1":
    update_yaml("lightgbm.yaml", lambda p: p.update({"device_type": "gpu"}))

print(f"Wrote GPU-aware model configs to {target}")
'@
Invoke-LoggedPythonCode -Name "write_gpu_model_configs" -Code $gpuConfigCode -Arguments @($script:CONFIG_DIR)

if ($script:RUN_PYTEST -eq 1) {
  Invoke-LoggedCommand -Name "pytest_preflight" -Command @(
    $script:PYTHON
    "-m"
    "pytest"
    "-W"
    "ignore::requests.exceptions.RequestsDependencyWarning"
    "--ignore"
    "results"
    "--basetemp"
    $script:PYTEST_BASE_TEMP_DIR
    "-o"
    "cache_dir=$($script:PYTEST_CACHE_DIR)"
  )
}

if ($script:SKIP_PREPROCESS -ne 1) {
  Run-StepIfMissing -Name "cohort_selection" -RequiredFiles @($script:COHORT_PATH, $script:COHORT_IDS_PATH) -Command @(
    $script:PYTHON
    "src/cohort_selection.py"
  )
  Run-StepIfMissing -Name "feature_extraction_baseline" -RequiredFiles @($script:BASELINE_FEATURES_PATH, $script:BASELINE_FEATURE_STATS_PATH) -Command @(
    $script:PYTHON
    "src/feature_extraction.py"
    "--disable-expanded-features"
  )
  if (-not (Validate-CsvColumnsAndRows -Path $script:BASELINE_FEATURES_PATH -Description "baseline extracted features" -RequiredColumns @("subject_id", "hadm_id", "stay_id", "mortality"))) {
    if ($script:SKIP_PREPROCESS -eq 1) {
      exit 1
    }
    Write-RunLog "Regenerating baseline extracted features due to invalid cached output."
    Invoke-LoggedCommand -Name "feature_extraction_baseline_repair" -Command @(
      $script:PYTHON
      "src/feature_extraction.py"
      "--disable-expanded-features"
    )
  }

  Run-StepIfMissing -Name "feature_extraction_expanded" -RequiredFiles @($script:EXPANDED_FEATURES_PATH, $script:EXPANDED_FEATURE_STATS_PATH) -Command @(
    $script:PYTHON
    "src/feature_extraction.py"
    "--enable-expanded-features"
  )
  if (-not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_FEATURES_PATH -Description "expanded extracted features" -RequiredColumns @("subject_id", "hadm_id", "stay_id", "mortality"))) {
    if ($script:SKIP_PREPROCESS -eq 1) {
      exit 1
    }
    Write-RunLog "Regenerating expanded extracted features due to invalid cached output."
    Invoke-LoggedCommand -Name "feature_extraction_expanded_repair" -Command @(
      $script:PYTHON
      "src/feature_extraction.py"
      "--enable-expanded-features"
    )
  }

  Run-StepIfMissing -Name "preprocess_baseline_all" -RequiredFiles @($script:BASELINE_XGBOOST_PATH) -Command (Add-PreprocessArgs @(
    $script:PYTHON
    "src/data_preprocessing.py"
    "--input-path"
    $script:BASELINE_FEATURES_PATH
    "--model-type"
    "xgboost"
    "--output-path"
    $script:BASELINE_XGBOOST_PATH
    "--report-dir"
    (Join-Path $script:RESULTS_ROOT "preprocess_reports/baseline")
  ))
  $preprocStatus = Ensure-PreprocessedDataset -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL)
  if ($preprocStatus -eq 2) {
    Invoke-LoggedCommand -Name "preprocess_baseline_all_repair" -Command (Add-PreprocessArgs @(
      $script:PYTHON
      "src/data_preprocessing.py"
      "--input-path"
      $script:BASELINE_FEATURES_PATH
      "--model-type"
      "xgboost"
      "--output-path"
      $script:BASELINE_XGBOOST_PATH
      "--report-dir"
      (Join-Path $script:RESULTS_ROOT "preprocess_reports/baseline")
    ))
  }
  elseif ($preprocStatus -ne 0) {
    exit 1
  }

  Run-StepIfMissing -Name "preprocess_expanded_xgboost" -RequiredFiles @($script:EXPANDED_XGBOOST_PATH) -Command (Add-PreprocessArgs @(
    $script:PYTHON
    "src/data_preprocessing.py"
    "--input-path"
    $script:EXPANDED_FEATURES_PATH
    "--output-path"
    $script:EXPANDED_XGBOOST_PATH
    "--model-type"
    "xgboost"
    "--report-dir"
    (Join-Path $script:RESULTS_ROOT "preprocess_reports/expanded_xgboost")
  ))
  $preprocStatus = Ensure-PreprocessedDataset -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL)
  if ($preprocStatus -eq 2) {
    Invoke-LoggedCommand -Name "preprocess_expanded_xgboost_repair" -Command (Add-PreprocessArgs @(
      $script:PYTHON
      "src/data_preprocessing.py"
      "--input-path"
      $script:EXPANDED_FEATURES_PATH
      "--output-path"
      $script:EXPANDED_XGBOOST_PATH
      "--model-type"
      "xgboost"
      "--report-dir"
      (Join-Path $script:RESULTS_ROOT "preprocess_reports/expanded_xgboost")
    ))
  }
  elseif ($preprocStatus -ne 0) {
    exit 1
  }
}
else {
  Write-RunLog "Skipping preprocessing because SKIP_PREPROCESS=1"
}

if ($script:RUN_MODEL_SUITE -eq 1) {
  Require-File -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBOOST features"
  if (-not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
    if ($script:SKIP_PREPROCESS -eq 1) {
      Write-RunLog "SKIP_PREPROCESS=1: cannot repair missing target column."
      exit 1
    }
    Write-RunLog "Repairing expanded preprocessed data before model suite."
    Invoke-LoggedCommand -Name "preprocess_expanded_xgboost_repair" -Command (Add-PreprocessArgs @(
      $script:PYTHON
      "src/data_preprocessing.py"
      "--input-path"
      $script:EXPANDED_FEATURES_PATH
      "--output-path"
      $script:EXPANDED_XGBOOST_PATH
      "--model-type"
      "xgboost"
      "--report-dir"
      (Join-Path $script:RESULTS_ROOT "preprocess_reports/expanded_xgboost")
    ))
    if (-not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
      exit 1
    }
  }

  Run-StepIfMissing -Name "model_suite_gpu" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "model_suite/model_suite_results.csv")
    (Join-Path $script:RESULTS_ROOT "model_suite/model_comparison_table.csv")
  ) -Command @(
    $script:PYTHON
    "src/experiments/run_model_suite.py"
    "--data-path"
    $script:EXPANDED_XGBOOST_PATH
    "--target-col"
    $script:TARGET_COL
    "--output-dir"
    (Join-Path $script:RESULTS_ROOT "model_suite")
    "--config-dir"
    $script:CONFIG_DIR
    "--seed"
    $script:SEED
    "--bootstrap-iterations"
    $script:BOOTSTRAP_ITERATIONS
  )
}

if ($script:RUN_XGBOOST_ABLATION -eq 1) {
  Require-File -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features"
  Require-File -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features"

  if (
    -not (Validate-CsvColumnsAndRows -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL)) -or
    -not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))
  ) {
    if ($script:SKIP_PREPROCESS -eq 1) {
      Write-RunLog "SKIP_PREPROCESS=1: cannot repair missing target column(s) in cached files."
      exit 1
    }

    if (-not (Validate-CsvColumnsAndRows -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
      Write-RunLog "Repairing baseline preprocessed data before ablation."
      Invoke-LoggedCommand -Name "preprocess_baseline_all_repair" -Command (Add-PreprocessArgs @(
        $script:PYTHON
        "src/data_preprocessing.py"
        "--input-path"
        $script:BASELINE_FEATURES_PATH
        "--model-type"
        "xgboost"
        "--output-path"
        $script:BASELINE_XGBOOST_PATH
        "--report-dir"
        (Join-Path $script:RESULTS_ROOT "preprocess_reports/baseline")
      ))
      if (-not (Validate-CsvColumnsAndRows -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBOOST features" -RequiredColumns @($script:TARGET_COL))) {
        exit 1
      }
    }

    if (-not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
      Write-RunLog "Repairing expanded preprocessed data before ablation."
      Invoke-LoggedCommand -Name "preprocess_expanded_xgboost_repair" -Command (Add-PreprocessArgs @(
        $script:PYTHON
        "src/data_preprocessing.py"
        "--input-path"
        $script:EXPANDED_FEATURES_PATH
        "--output-path"
        $script:EXPANDED_XGBOOST_PATH
        "--model-type"
        "xgboost"
        "--report-dir"
        (Join-Path $script:RESULTS_ROOT "preprocess_reports/expanded_xgboost")
      ))
      if (-not (Validate-CsvColumnsAndRows -Path $script:EXPANDED_XGBOOST_PATH -Description "expanded preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
        exit 1
      }
    }
  }

  Run-StepIfMissing -Name "xgboost_baseline_vs_expanded_cuda" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "xgboost_expanded_ablation/ablation_threshold_policy_results.csv")
  ) -Command @(
    $script:PYTHON
    "tools/run_xgboost_ablation.py"
    "--baseline-data-path"
    $script:BASELINE_XGBOOST_PATH
    "--expanded-data-path"
    $script:EXPANDED_XGBOOST_PATH
    "--output-dir"
    (Join-Path $script:RESULTS_ROOT "xgboost_expanded_ablation")
    "--device"
    "cuda"
    "--n-estimators"
    $script:XGBOOST_ABLATION_ESTIMATORS
    "--early-stopping-rounds"
    $script:XGBOOST_EARLY_STOPPING_ROUNDS
  )
}

if ($script:RUN_LEGACY_XGBOOST_ENSEMBLE -eq 1) {
  Require-File -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features"
  if (-not (Validate-CsvColumnsAndRows -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBoost features" -RequiredColumns @($script:TARGET_COL))) {
    if ($script:SKIP_PREPROCESS -eq 1) {
      Write-RunLog "SKIP_PREPROCESS=1: cannot repair missing target column in baseline features."
      exit 1
    }
    Write-RunLog "Repairing baseline preprocessed data before legacy ensemble."
    Invoke-LoggedCommand -Name "preprocess_baseline_all_repair" -Command (Add-PreprocessArgs @(
      $script:PYTHON
      "src/data_preprocessing.py"
      "--input-path"
      $script:BASELINE_FEATURES_PATH
      "--model-type"
      "xgboost"
      "--output-path"
      $script:BASELINE_XGBOOST_PATH
      "--report-dir"
      (Join-Path $script:RESULTS_ROOT "preprocess_reports/baseline")
    ))
    if (-not (Validate-CsvColumnsAndRows -Path $script:BASELINE_XGBOOST_PATH -Description "baseline preprocessed XGBOOST features" -RequiredColumns @($script:TARGET_COL))) {
      exit 1
    }
  }
  Run-StepIfMissing -Name "legacy_xgboost_ensemble_cuda" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "legacy_xgboost_ensemble/xgboost/ensemble_threshold_results.csv")
  ) -Command @(
    $script:PYTHON
    "src/main.py"
    "--model"
    "xgboost_ensemble"
    "--data-path"
    $script:BASELINE_XGBOOST_PATH
    "--output-dir"
    (Join-Path $script:RESULTS_ROOT "legacy_xgboost_ensemble")
    "--ensemble-size"
    $script:LEGACY_XGBOOST_ENSEMBLE_SIZE
    "--no-tune"
  )
}

if (($script:RUN_MAFNET -eq 1) -or ($script:RUN_MAFNET_ABLATIONS -eq 1)) {
  Require-File -Path $script:MAFNET_COHORT_PATH -Description "MAFNet cohort CSV"
  Require-File -Path $script:MAFNET_FEATURES_PATH -Description "MAFNet feature CSV"
  Run-StepIfMissing -Name "build_mafnet_first6h_events" -RequiredFiles @(
    $script:MAFNET_EVENTS_PATH
  ) -Command @(
    $script:PYTHON
    "tools/build_first6h_events.py"
    "--cohort-path"
    $script:MAFNET_COHORT_PATH
    "--icu-dir"
    "data/icu"
    "--hosp-dir"
    "data/hosp"
    "--output-path"
    $script:MAFNET_EVENTS_PATH
  )
}

if ($script:RUN_MAFNET -eq 1) {
  Run-StepIfMissing -Name "mafnet_full_cuda" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "mafnet_full/validation_metrics.json")
    (Join-Path $script:RESULTS_ROOT "mafnet_full/test_metrics.json")
  ) -Command @(
    $script:PYTHON
    "src/training/train_mafnet.py"
    "--events-path"
    $script:MAFNET_EVENTS_PATH
    "--cohort-path"
    $script:MAFNET_COHORT_PATH
    "--features-path"
    $script:MAFNET_FEATURES_PATH
    "--config"
    "configs/mafnet.yaml"
    "--output-dir"
    (Join-Path $script:RESULTS_ROOT "mafnet_full")
    "--device"
    "cuda"
    "--evaluate-test"
  )
}

if ($script:RUN_MAFNET_ABLATIONS -eq 1) {
  Run-StepIfMissing -Name "mafnet_ablations_cuda" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "mafnet_ablations/mafnet_ablation_results.csv")
  ) -Command @(
    $script:PYTHON
    "src/experiments/run_mafnet_ablations.py"
    "--events-path"
    $script:MAFNET_EVENTS_PATH
    "--cohort-path"
    $script:MAFNET_COHORT_PATH
    "--features-path"
    $script:MAFNET_FEATURES_PATH
    "--config"
    "configs/mafnet.yaml"
    "--output-dir"
    (Join-Path $script:RESULTS_ROOT "mafnet_ablations")
    "--device"
    "cuda"
    "--evaluate-test"
  )
}

if ($script:RUN_FINAL_SUMMARY -eq 1 -and (Test-Path (Join-Path $script:RESULTS_ROOT "model_suite"))) {
  Run-StepIfMissing -Name "final_summary" -RequiredFiles @(
    (Join-Path $script:RESULTS_ROOT "final_summary.md")
  ) -Command @(
    $script:PYTHON
    "src/experiments/generate_final_summary.py"
    "--model-suite-dir"
    (Join-Path $script:RESULTS_ROOT "model_suite")
    "--output-path"
    (Join-Path $script:RESULTS_ROOT "final_summary.md")
  )
}

$collectCode = @'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

results_root = Path(sys.argv[1])
run_name = sys.argv[2]
records: list[dict] = []


def add_csv(stage: str, path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df.insert(0, "run_name", run_name)
    df.insert(1, "stage", stage)
    df.insert(2, "source_file", str(path))
    records.extend(df.to_dict(orient="records"))


add_csv("model_suite_results", results_root / "model_suite" / "model_suite_results.csv")
add_csv("model_suite_comparison", results_root / "model_suite" / "model_comparison_table.csv")
add_csv(
    "xgboost_expanded_ablation",
    results_root / "xgboost_expanded_ablation" / "ablation_threshold_policy_results.csv",
)
add_csv("mafnet_ablations", results_root / "mafnet_ablations" / "mafnet_ablation_results.csv")
add_csv("legacy_xgboost_ensemble", results_root / "legacy_xgboost_ensemble" / "ensemble_threshold_results.csv")

for name in ["validation_metrics.json", "test_metrics.json"]:
    path = results_root / "mafnet_full" / name
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "run_name": run_name,
                "stage": f"mafnet_full_{path.stem}",
                "source_file": str(path),
                "model_name": "mafnet",
            }
        )
        records.append(payload)

combined = pd.DataFrame(records)
combined_path = results_root / "all_training_results.csv"
combined.to_csv(combined_path, index=False)

index_rows = []
for csv_path in sorted(results_root.rglob("*.csv")):
    if csv_path.name == "all_training_results.csv":
        continue
    index_rows.append({"run_name": run_name, "csv_file": str(csv_path)})
pd.DataFrame(index_rows).to_csv(results_root / "result_csv_index.csv", index=False)

manifest = {
    "run_name": run_name,
    "results_root": str(results_root),
    "combined_results_csv": str(combined_path),
    "result_csv_index": str(results_root / "result_csv_index.csv"),
    "n_combined_rows": int(len(combined)),
}
(results_root / "overnight_manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=True),
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2))
'@
Invoke-LoggedPythonCode -Name "collect_training_results" -Code $collectCode -Arguments @($script:RESULTS_ROOT, $runName)

Write-RunLog "Overnight training run finished."
Write-RunLog "Combined comparison CSV: $($script:RESULTS_ROOT)/all_training_results.csv"
Write-RunLog "CSV index: $($script:RESULTS_ROOT)/result_csv_index.csv"
