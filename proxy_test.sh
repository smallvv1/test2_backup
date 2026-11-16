echo "===== 当前代理变量 ====="
env | grep -i proxy || echo "没有检测到代理变量"

echo "===== 测试 GitHub ====="
curl -I --max-time 15 https://github.com

echo "===== 测试 OpenAI ====="
curl -I --max-time 15 https://chat.openai.com || true
curl -I --max-time 15 https://api.openai.com || true
