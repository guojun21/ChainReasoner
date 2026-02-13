# macOS iCloud Dataless 文件恢复方案

> 2026-02-12 实战总结。文件夹在 iCloud 同步的 Documents 目录下搬家后，大量文件变成 APFS dataless 空壳（元数据在、内容不在），导致 Python import 卡死、git 命令卡死、cat/cp/wc 全部挂起。

---

## 一、什么是 dataless

macOS Sonoma 起，iCloud 不再用 .icloud 占位符，改用 APFS dataless 机制：

| 部分 | 状态 |
|------|------|
| 文件名、大小、权限、时间戳 | 正常（ls -la 看不出异常） |
| 文件实际内容（字节数据） | 不在本地，blocks=0 |

检测方法：
```bash
stat -f "%Sf %b %N" <file>
# 输出 "compressed,dataless 0 xxx" = 空壳
# 输出 "- 8 xxx" = 正常
```

批量检测：
```bash
find /path/to/dir -path "./.venv" -prune -o -type f -print0 | \
  xargs -0 stat -f "%Sf|%N" 2>/dev/null | grep "dataless"
```

---

## 二、为什么会发生

三因素叠加：
1. Documents 文件夹开启了 iCloud 同步 + "优化 Mac 储存空间"开着
2. 在 Finder 里拖动/重命名文件夹 -> iCloud 检测到变动 -> 重新同步
3. 同步后 iCloud 认为文件已安全存在云端 -> 把本地内容清掉变成 dataless 空壳

---

## 三、恢复方案（按优先级）

### 方案 A：从 git 恢复（最快最可靠）

```bash
git show HEAD:path/to/file > /tmp/_restored.py
rm -f path/to/file
mv /tmp/_restored.py path/to/file
```

批量：
```bash
find . -path "./.venv" -prune -o -path "./.git" -prune -o -type f -print0 | \
  xargs -0 stat -f "%Sf|%N" 2>/dev/null | grep "dataless" | cut -d'|' -f2 | \
  while read f; do
    rel="${f#./}"
    if git show "HEAD:$rel" > /tmp/_tmp 2>/dev/null; then
      rm -f "$f"; mv /tmp/_tmp "$f"; echo "OK: $rel"
    else
      echo "SKIP: $rel"
    fi
  done
```

### 方案 A-2：.git 也坏了 -> /tmp 重新 clone

```bash
git clone <url> /tmp/repo_clean
cp local/file /tmp/repo_clean/same/path/
cd /tmp/repo_clean && git add -A && git commit -m "msg" && git push
rm -rf /project/.git && cp -R /tmp/repo_clean/.git /project/.git
```

### 方案 B：已知内容直接写入

```bash
cat > /tmp/_new.py << 'EOF'
内容
EOF
rm -f /path/to/old.py
mv /tmp/_new.py /path/to/old.py
```

### 方案 C：Finder 打开或右键"立即下载"（碰运气）

### 方案 D：brctl download（Sonoma 后效果不稳定）

---

## 四、.venv 恢复

```bash
rm -rf .venv && python3 -m venv .venv && pip install -r requirements.txt
```

## 五、__pycache__ 问题

```bash
find . -path "./.venv" -prune -o -type d -name "__pycache__" -print -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 python3 -B your_script.py
```

## 六、根本预防

1. 关闭"优化 Mac 储存空间"
2. 代码项目不放在 iCloud 同步目录（Documents）下
3. 定期 git push
