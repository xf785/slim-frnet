' ============================================================
'  synthfp - 无控制台窗口启动图形界面（Windows）
' ============================================================
'  用途：由 start_gui.bat 或资源管理器调用，用 pythonw.exe 启动
'  `python -m synthfp gui`，不留下黑色命令行窗口。
'
'  直接双击本文件即可；也可：
'      wscript tools\launch_gui.vbs
' ============================================================

Option Explicit

Dim shell, fso, here, parent, candidates, i, pyw, cmd

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)   ' ...\tools
parent = fso.GetParentFolderName(here)                   ' 项目根目录

' 按优先级找一个可用的 pythonw.exe
candidates = Array( _
    parent & "\.venv\Scripts\pythonw.exe", _
    parent & "\venv\Scripts\pythonw.exe", _
    parent & "\env\Scripts\pythonw.exe", _
    shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python313\pythonw.exe", _
    shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python312\pythonw.exe", _
    shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python311\pythonw.exe" _
)

pyw = ""
For i = 0 To UBound(candidates)
    If pyw = "" Then
        If fso.FileExists(candidates(i)) Then pyw = candidates(i)
    End If
Next

If pyw = "" Then
    MsgBox "找不到 pythonw.exe。" & vbCrLf & vbCrLf & _
           "请先安装 Python 3.10+ 并执行：" & vbCrLf & _
           "    python -m pip install -r requirements.txt" & vbCrLf & vbCrLf & _
           "或直接双击项目根目录的 start_gui.bat。", _
           vbExclamation, "synthfp"
    WScript.Quit 1
End If

' 工作目录必须是项目根目录，才能 import 到 synthfp 包
shell.CurrentDirectory = parent

cmd = """" & pyw & """" & " -m synthfp gui"

On Error Resume Next
shell.Run cmd, 0, False      ' 0 = 隐藏窗口，False = 不等待
If Err.Number <> 0 Then
    MsgBox "启动失败：" & vbCrLf & Err.Description & vbCrLf & vbCrLf & _
           "可改用 start_gui.bat，它会显示错误信息。", _
           vbCritical, "synthfp"
    WScript.Quit 1
End If
On Error GoTo 0

WScript.Quit 0
