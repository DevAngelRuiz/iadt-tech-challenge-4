# Fala sintetica local, sem pacientes reais nem chamadas a servicos externos.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$sampleRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\data\samples'))
$samplePath = Join-Path $sampleRoot 'consulta_falada_sintetica.wav'
$referencePath = Join-Path $sampleRoot 'consulta_falada_sintetica.txt'
$sampleText = 'Estou muito cansada e sem energia. Tenho dificuldade para dormir. Sinto falta de ar quando caminho. Gostaria de conversar com a equipe de atendimento.'
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $portuguese = @($synthesizer.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'pt-BR' })
    if ($portuguese.Count -eq 0) {
        throw 'Nenhuma voz pt-BR instalada. Grave uma consulta ficticia autorizada e use esse arquivo na demonstracao.'
    }
    $synthesizer.SelectVoice($portuguese[0].VoiceInfo.Name)
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo -ArgumentList 16000,([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen),([System.Speech.AudioFormat.AudioChannel]::Mono)
    $synthesizer.SetOutputToWaveFile($samplePath, $format)
    $synthesizer.Speak($sampleText)
    $synthesizer.SetOutputToNull()
    Set-Content -LiteralPath $referencePath -Value $sampleText -Encoding UTF8
    [PSCustomObject]@{Arquivo=$samplePath;Voz=$portuguese[0].VoiceInfo.Name;Tipo='fala sintetica pt-BR'} | ConvertTo-Json -Compress
} finally {
    $synthesizer.Dispose()
}
