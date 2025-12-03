## 1. 修改 `self_play_trainer.py`
- **位置**: 第470-471行
- **修改内容**: 将性能记录频率从每200手牌改为每50手牌
- **修改前**: `if self.total_hands % 200 == 0:`
- **修改后**: `if self.total_hands % 50 == 0:`

## 2. 修改 `train.py`
- **位置**: 在训练器创建后添加verbose处理逻辑
- **修改内容**: 添加日志级别调整代码，当--verbose参数启用时，设置日志级别为DEBUG
- **修改代码**: 在第151行后添加:
  ```python
  # 如果启用详细日志，更新日志级别
  if args.verbose:
      logger.setLevel(logging.DEBUG)
  ```

## 3. 提交和推送更改
- 使用git add添加修改的文件
- 使用git commit提交更改，添加描述性提交信息
- 使用git push将更改推送到远程仓库

## 4. 验证更改
- 确保修改后的代码符合Python语法规范
- 确保所有功能保持不变
- 确保日志打印频率增加到每50手牌