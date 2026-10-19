param(
    [Parameter(Mandatory=$true)][string]$KernelDirectory,
    [string]$Owner='rayyannmerchant'
)
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
$resolvedKernel=(Resolve-Path -LiteralPath $KernelDirectory).Path
$metadata=Get-Content -LiteralPath (Join-Path $resolvedKernel 'kernel-metadata.json') -Raw | ConvertFrom-Json
if ($metadata.is_private -ne $true) { throw 'Kernel must be private' }
if ($metadata.id -ne "$Owner/upar2027-siglip2-semanticpar") { throw 'Unexpected kernel owner/slug' }
foreach ($source in $metadata.dataset_sources) {
    & kaggle datasets files $source
    if ($LASTEXITCODE -ne 0) { throw "Dataset source inaccessible: $source. No GPU kernel was launched." }
}
& kaggle quota
if ($LASTEXITCODE -ne 0) { throw 'Quota check failed' }
& kaggle kernels push -p $resolvedKernel --accelerator NvidiaTeslaT4 -t 43200
if ($LASTEXITCODE -ne 0) { throw 'Kernel push failed' }
Write-Output 'Inspect CLI output for omitted data-source warnings; exit 0 alone does not prove source attachment.'
