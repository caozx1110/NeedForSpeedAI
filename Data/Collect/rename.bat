setlocal EnableDelayedExpansion

FOR /f "tokens=*" %%i IN ('dir /a:d /b') DO (

ren "%%i" "%%i_3"

)