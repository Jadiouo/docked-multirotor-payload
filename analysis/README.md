執行方式（在專案根目錄）：`.venv/bin/python -m analysis.controllability`、`... -m analysis.identifiability`、`... -m analysis.closed_loop`。
輸出都寫到 results/p0/；順序無關，但 closed_loop 與 identifiability 會重用 controllability 的 LP 函式。
測試：`.venv/bin/python -m pytest -q`。identifiability 約 25 秒。
