#include "..\..\release\reuse-constants.iss"
#define AppVersion "0.4.3"
[Setup]
AppId={{718C23FC-78B1-471E-8F29-A47A9BA97FD1}
AppName=Jev Mahjong - Jev Play v2
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
Source: "..\..\release\reuse-runtime.sha256"; Flags: dontcopy
Source: "..\..\release\reuse-app.sha256"; Flags: dontcopy
Source: "..\..\release\reuse-model.sha256"; Flags: dontcopy
#include "..\..\release\runtime-files.iss"
[Icons]
Name: "{group}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
Name: "{autodesktop}\Jev Mahjong"; Filename: "{app}\JevMahjong.exe"
[Run]
Filename: "{app}\JevMahjong.exe"; Description: "Launch Jev Mahjong"; Flags: nowait postinstall skipifsilent
[Code]
function RuntimeBaseUrl(Param: String): String;
begin
  Result := ExpandConstant('{param:RUNTIMEURL|https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.4.3}');
end;


function BaselineBaseUrl(Param: String): String;
begin
  Result := ExpandConstant('{param:BASELINEURL|https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.4.0}');
end;

var
  CheckedDirectory: String;
  BaselineChecked, BaselineNeeded: Boolean;
  ApplicationChecked, ApplicationNeeded: Boolean;
  ModelChecked, ModelNeeded: Boolean;

procedure CheckDirectory;
var
  Directory: String;
begin
  Directory := ExpandConstant('{app}');
  if Directory <> CheckedDirectory then begin
    CheckedDirectory := Directory;
    BaselineChecked := False;
    ApplicationChecked := False;
    ModelChecked := False;
  end;
end;

function MatchesFiles(ListName: String): Boolean;
var
  Lines: TArrayOfString;
  Index: Integer;
  RelativeName, FileName, Expected, Actual: String;
begin
  Result := False;
  ExtractTemporaryFile(ListName);
  if not LoadStringsFromFile(ExpandConstant('{tmp}\') + ListName, Lines) then
    RaiseException('Cannot read bundled component checksums');
  if GetArrayLength(Lines) = 0 then
    RaiseException('Empty component checksum list');
  for Index := 0 to GetArrayLength(Lines) - 1 do begin
    if Length(Lines[Index]) < 67 then
      RaiseException('Invalid bundled component checksum');
    Expected := Copy(Lines[Index], 1, 64);
    RelativeName := Copy(Lines[Index], 67, Length(Lines[Index]));
    StringChangeEx(RelativeName, '/', '\', True);
    FileName := AddBackslash(CheckedDirectory) + RelativeName;
    if not FileExists(FileName) then begin
      Log('Component file missing: ' + RelativeName);
      Exit;
    end;
    try
      Actual := GetSHA256OfFile(FileName);
    except
      Log('Cannot verify component file: ' + RelativeName);
      Exit;
    end;
    if CompareText(Actual, Expected) <> 0 then begin
      Log('Component file changed: ' + RelativeName);
      Exit;
    end;
  end;
  Result := True;
end;

function NeedBaseline: Boolean;
begin
  CheckDirectory;
  if not BaselineChecked then begin
    BaselineNeeded := not MatchesFiles('reuse-runtime.sha256');
    BaselineChecked := True;
    if BaselineNeeded then
      Log('JEV_UPDATE baseline=download')
    else
      Log('JEV_UPDATE baseline=reuse');
  end;
  Result := BaselineNeeded;
end;

function NeedApplication: Boolean;
begin
  CheckDirectory;
  if not ApplicationChecked then begin
    ApplicationNeeded := NeedBaseline;
    if not ApplicationNeeded then
      ApplicationNeeded := not MatchesFiles('reuse-app.sha256');
    ApplicationChecked := True;
    if ApplicationNeeded then
      Log('JEV_UPDATE application=download')
    else
      Log('JEV_UPDATE application=reuse');
  end;
  Result := ApplicationNeeded;
end;

function NeedModel: Boolean;
begin
  CheckDirectory;
  if not ModelChecked then begin
    if NeedBaseline then
      ModelNeeded := not {#ModelInBaseline}
    else
      ModelNeeded := not MatchesFiles('reuse-model.sha256');
    ModelChecked := True;
    if ModelNeeded then
      Log('JEV_UPDATE model=download')
    else
      Log('JEV_UPDATE model=reuse');
  end;
  Result := ModelNeeded;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine;
  if NeedBaseline then
    Result := Result + '首次安装或环境不完整：将下载并修复运行组件。' + NewLine
  else
    Result := Result + '现有运行环境已校验，将直接复用。' + NewLine;
  if NeedApplication then
    Result := Result + '将更新程序及发生变化的文件。' + NewLine;
  if NeedModel then
    Result := Result + '将下载模型更新。'
  else
    Result := Result + '模型无需重复下载。';
end;
