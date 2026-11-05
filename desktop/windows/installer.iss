#define AppVersion "0.2.0"
[Setup]
AppId={{718C23FC-78B1-471E-8F29-A47A9BA97FD1}
AppName=Jev Mahjong
AppVersion={#AppVersion}
AppPublisher=YokyYao
DefaultDirName={localappdata}\Programs\JevMahjong
DefaultGroupName=Jev Mahjong
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\release
OutputBaseFilename=JevMahjong-Setup-{#AppVersion}-win-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\JevMahjong.exe
LicenseFile=..\LICENSE
[Files]
Source: "..\..\dist\JevMahjong\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
Name: "{autodesktop}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
[Run]
Filename: "{app}\JevMahjong.exe"; Description: "Launch Jev Mahjong"; Flags: nowait postinstall skipifsilent
