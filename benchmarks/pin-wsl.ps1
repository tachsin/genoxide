<#
.SYNOPSIS
    Pins the WSL 2 virtual machine to chosen cores, e.g. the fastest P-cores of a hybrid Intel CPU,
    so that benchmark times don't depend on which core Windows picks.

.DESCRIPTION
    Without switches, it sets the processor affinity of the WSL virtual machine (the vmmemWSL
    process) once. That needs an Administrator PowerShell and a running WSL, and lasts until the
    virtual machine stops: `wsl --shutdown`, or about a minute after the last Linux process ends.

    -Install registers a scheduled task that pins it at logon and every minute, with the highest
    privileges, so it stays pinned. -Uninstall removes the task.

    benchmarks/run.py checks the pinning before it measures times, and refuses to measure without it.

.PARAMETER Cores
    The logical processors to pin to: 8 and 19 by default, the two favoured P-cores (the highest
    turbo frequency) of the Intel Core Ultra 7 265K the published benchmarks run on. Measured there,
    a fixed single-threaded C loop varied by about 1% on them, against 1.5% on all 8 P-cores and
    more when Windows can use the E-cores too.

.EXAMPLE
    .\pin-wsl.ps1 -Install
#>
param(
    [int[]]$Cores = @(8, 19),
    [switch]$Install,
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$task = 'genoxide benchmarks: pin WSL'

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $task -Confirm:$false
    "removed the scheduled task '$task'"
    return
}

if ($Install) {
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Cores $($Cores -join ',')"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
    # at logon, then every minute: WSL's virtual machine restarts whenever WSL starts again
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1)).Repetition
    # S4U: runs without a window, whether or not the user is logged on
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $task
    "installed the scheduled task '$task': it pins WSL to cores $($Cores -join ', ') at logon and every minute"
    return
}

$mask = [int64]0
foreach ($core in $Cores) {
    $mask = $mask -bor ([int64]1 -shl $core)
}
$vm = Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vm) {
    'WSL is not running: nothing to pin'
    return
}
try {
    if ([int64]$vm.ProcessorAffinity -ne $mask) {
        $vm.ProcessorAffinity = [IntPtr]$mask
    }
} catch {
    throw 'pinning WSL needs an Administrator PowerShell (or the scheduled task of -Install)'
}
'WSL (process {0}) pinned to cores {1} (mask 0x{2:X})' -f $vm.Id, ($Cores -join ', '), $mask
