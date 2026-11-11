#define AppVersion "0.3.0"
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
Compression=lzma2/normal
SolidCompression=yes
ArchiveExtraction=full
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\JevMahjong.exe
LicenseFile=..\LICENSE
[Files]
#include "..\..\release\runtime-files.iss"
[Icons]
Name: "{group}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
Name: "{autodesktop}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
[Run]
Filename: "{app}\JevMahjong.exe"; Description: "Launch Jev Mahjong"; Flags: nowait postinstall skipifsilent
[Code]
function RuntimeBaseUrl(Param: String): String;
begin
  Result := ExpandConstant('{param:RUNTIMEURL|https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.3.0}');
end;
