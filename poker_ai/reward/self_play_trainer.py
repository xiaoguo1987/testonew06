# 德州扑克自我对弈训练器

import os
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SelfPlayTrainer')

from ..game.poker_game import PokerGame, Player
from ..model.ai_agent import PokerAIAgent, EnsembleAgent
from ..reward.reward_calculator import RewardCalculator, AdaptiveRewardCalculator

class SelfPlayTrainer:
    """德州扑克自我对弈训练器"""
    
    def __init__(self,
                 model_type: str = 'dqn',  # 'dqn' 或 'policy_gradient'
                 checkpoint_dir: str = './checkpoints',
                 small_blind: int = 5,
                 big_blind: int = 10,
                 initial_stack: int = 1000,
                 num_players: int = 6,
                 eval_interval: int = 100,  # 评估间隔（局数）
                 checkpoint_interval: int = 500,  # 保存检查点间隔（局数）
                 num_eval_games: int = 50,  # 每次评估的游戏局数
                 use_ensemble: bool = False):
        
        self.model_type = model_type
        self.checkpoint_dir = checkpoint_dir
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        self.num_players = num_players
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.num_eval_games = num_eval_games
        self.use_ensemble = use_ensemble
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建AI代理
        self._create_agents()
        
        # 创建奖励计算器
        self.reward_calculator = AdaptiveRewardCalculator()
        
        # 统计信息
        self.total_hands = 0
        self.total_profit = 0
        self.best_score = -float('inf')
        self.performance_history = []
    
    def _create_agents(self) -> None:
        """创建AI代理"""
        if self.use_ensemble:
            self.main_agent = EnsembleAgent()
        else:
            self.main_agent = PokerAIAgent(agent_type=self.model_type)
        
        # 创建对手代理（使用历史版本）
        self.opponent_agents = []
        for i in range(self.num_players - 1):  # 假设主代理始终是第一个玩家
            if self.use_ensemble:
                self.opponent_agents.append(EnsembleAgent())
            else:
                self.opponent_agents.append(PokerAIAgent(agent_type=self.model_type))
    
    def _create_game(self) -> Tuple[PokerGame, List[Player]]:
        """创建游戏实例和玩家"""
        # 创建玩家
        players = []
        
        # 主玩家（我们训练的AI）
        main_player = Player("Main_AI", self.initial_stack, True)
        players.append(main_player)
        
        # 对手玩家
        for i in range(self.num_players - 1):
            opponent = Player(f"Opponent_{i+1}", self.initial_stack, True)
            players.append(opponent)
        
        # 创建游戏
        game = PokerGame(self.small_blind, self.big_blind, players)
        
        return game, players
    
    def _play_hand(self, game: PokerGame, players: List[Player], is_training: bool = True) -> Dict:
        """
        玩一手牌
        返回: 包含游戏结果的字典
        """
        # 记录每个玩家的动作历史
        action_histories = {player.name: [] for player in players}
        
        # 开始新的一手牌
        game.start_new_hand()
        
        # 游戏主循环
        while True:
            # 处理当前轮次的所有动作
            while not game.is_round_over():
                current_player = game.next_player()
                if not current_player:
                    break
                
                # 决定使用哪个代理
                if current_player.name == "Main_AI":
                    agent = self.main_agent
                else:
                    # 为对手选择一个代理
                    opponent_idx = int(current_player.name.split('_')[-1]) - 1
                    if opponent_idx < len(self.opponent_agents):
                        agent = self.opponent_agents[opponent_idx]
                    else:
                        # 如果索引无效，使用第一个对手代理
                        agent = self.opponent_agents[0]
                
                # 选择动作
                action, amount = agent.select_action(game, current_player)
                
                # 记录动作
                action_histories[current_player.name].append((action, amount))
                
                # 执行动作
                game.handle_action(current_player, action, amount)
            
            # 进入下一轮或结束游戏
            if game.proceed_to_next_round():
                break
        
        # 计算结果
        result = {
            'winner': None,
            'player_profits': {},
            'action_histories': action_histories
        }
        
        # 确定获胜者和利润
        # 注意：在实际的PokerGame类中，我们需要修改_showdown方法来记录获胜者
        # 这里简化处理，假设筹码最多的玩家是获胜者
        winner = max(players, key=lambda p: p.stack)
        result['winner'] = winner.name
        
        # 计算每个玩家的利润
        for player in players:
            # 初始筹码减去当前筹码的相反数即为利润
            profit = player.stack - self.initial_stack
            result['player_profits'][player.name] = profit
        
        # 如果是训练模式，更新代理
        if is_training:
            main_player = players[0]
            main_profit = result['player_profits'][main_player.name]
            is_main_winner = result['winner'] == main_player.name
            
            # 为每个动作计算奖励并更新代理
            for action, amount in action_histories[main_player.name]:
                # 计算奖励
                reward = self.reward_calculator.calculate_reward(
                    main_player, game, action, amount, 
                    is_winner=is_main_winner, profit_loss=main_profit
                )
                
                # 更新代理
                # 注意：这里简化处理，实际应该传入完整的状态转换
                self.main_agent.learn(reward, done=False)
            
            # 回合结束时的最终更新
            final_reward = self.reward_calculator.calculate_reward(
                main_player, game, 
                action_histories[main_player.name][-1][0] if action_histories[main_player.name] else 'fold',
                action_histories[main_player.name][-1][1] if action_histories[main_player.name] else None,
                is_winner=is_main_winner, profit_loss=main_profit
            )
            self.main_agent.learn(final_reward, done=True)
        
        return result
    
    def evaluate(self) -> Dict:
        """
        评估当前模型的性能
        返回: 评估结果
        """
        logger.info(f"开始评估模型，共进行 {self.num_eval_games} 局游戏")
        
        total_profit = 0
        wins = 0
        
        for i in range(self.num_eval_games):
            game, players = self._create_game()
            result = self._play_hand(game, players, is_training=False)
            
            # 记录结果
            main_profit = result['player_profits']['Main_AI']
            total_profit += main_profit
            if result['winner'] == 'Main_AI':
                wins += 1
        
        # 计算统计信息
        avg_profit = total_profit / self.num_eval_games
        win_rate = wins / self.num_eval_games
        
        eval_result = {
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'wins': wins,
            'total_hands': self.num_eval_games
        }
        
        logger.info(f"评估完成: 平均利润={avg_profit:.2f}, 胜率={win_rate:.2%}")
        
        # 更新性能历史
        self.performance_history.append(eval_result)
        
        return eval_result
    
    def update_opponents(self) -> None:
        """
        更新对手代理，使用主代理的历史版本
        """
        # 我们可以随机选择一些历史检查点作为对手
        # 这里简化处理，只是复制当前主代理的一部分权重
        
        # 随机选择一个对手进行替换
        if self.opponent_agents:
            opponent_idx = random.randint(0, len(self.opponent_agents) - 1)
            
            # 保存当前主代理的权重
            temp_path = os.path.join(self.checkpoint_dir, 'temp_agent.h5')
            self.main_agent.save_model(temp_path)
            
            # 加载到对手代理
            try:
                self.opponent_agents[opponent_idx].load_model(temp_path)
                logger.info(f"更新了对手 {opponent_idx + 1} 的模型")
            except Exception as e:
                logger.error(f"更新对手模型失败: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        保存模型检查点
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"best_agent_{self.model_type}_{timestamp}.h5"
        else:
            filename = f"agent_{self.model_type}_{self.total_hands}.h5"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            self.main_agent.save_model(filepath)
            logger.info(f"保存模型检查点: {filename}")
            
            # 保存训练状态
            state_filepath = filepath.replace('.h5', '_state.pkl')
            with open(state_filepath, 'wb') as f:
                pickle.dump({
                    'total_hands': self.total_hands,
                    'total_profit': self.total_profit,
                    'best_score': self.best_score,
                    'performance_history': self.performance_history
                }, f)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, filepath: Optional[str] = None) -> bool:
        """
        加载模型检查点
        """
        if filepath is None:
            # 尝试加载最佳模型
            best_models = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('best_agent_')]
            if best_models:
                # 选择最新的最佳模型
                best_models.sort(reverse=True)
                filepath = os.path.join(self.checkpoint_dir, best_models[0])
            else:
                # 尝试加载最新的检查点
                checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('agent_') and f.endswith('.h5')]
                if checkpoints:
                    checkpoints.sort(reverse=True, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    filepath = os.path.join(self.checkpoint_dir, checkpoints[0])
                else:
                    logger.warning("没有找到检查点文件")
                    return False
        
        try:
            self.main_agent.load_model(filepath)
            logger.info(f"加载模型检查点: {filepath}")
            
            # 尝试加载训练状态
            state_filepath = filepath.replace('.h5', '_state.pkl')
            if os.path.exists(state_filepath):
                with open(state_filepath, 'rb') as f:
                    state = pickle.load(f)
                    self.total_hands = state.get('total_hands', 0)
                    self.total_profit = state.get('total_profit', 0)
                    self.best_score = state.get('best_score', -float('inf'))
                    self.performance_history = state.get('performance_history', [])
            
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False
    
    def train(self, num_hands: int) -> None:
        """
        训练模型
        num_hands: 要训练的手牌数量
        """
        logger.info(f"开始训练模型，共训练 {num_hands} 手牌")
        
        # 尝试加载现有检查点
        self.load_checkpoint()
        
        start_hand = self.total_hands
        
        for hand_idx in range(start_hand, start_hand + num_hands):
            # 创建新游戏
            game, players = self._create_game()
            
            # 玩一手牌
            result = self._play_hand(game, players, is_training=True)
            
            # 更新统计信息
            self.total_hands += 1
            self.total_profit += result['player_profits']['Main_AI']
            
            # 每100手牌打印进度
            if self.total_hands % 100 == 0:
                logger.info(f"已完成 {self.total_hands} 手牌，累计利润: {self.total_profit:.2f}")
            
            # 定期评估
            if self.total_hands % self.eval_interval == 0:
                eval_result = self.evaluate()
                
                # 更新奖励计算器
                self.reward_calculator.record_performance(eval_result['total_profit'], self.num_eval_games)
                self.reward_calculator.adjust_rewards()
                
                # 如果表现更好，保存为最佳模型
                current_score = eval_result['avg_profit'] * eval_result['win_rate']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.save_checkpoint(is_best=True)
                    logger.info(f"新的最佳模型！得分: {current_score:.4f}")
            
            # 定期更新对手
            if self.total_hands % (self.eval_interval // 2) == 0:
                self.update_opponents()
            
            # 定期保存检查点
            if self.total_hands % self.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
        
        logger.info(f"训练完成！共完成 {num_hands} 手牌训练")
        
        # 保存最终模型
        self.save_checkpoint(is_best=False)
        
        # 最后评估一次
        final_eval = self.evaluate()
        logger.info(f"最终评估结果: {final_eval}")

# 示例使用
if __name__ == "__main__":
    # 创建训练器
    trainer = SelfPlayTrainer(
        model_type='dqn',
        checkpoint_dir='./checkpoints',
        small_blind=5,
        big_blind=10,
        initial_stack=1000,
        eval_interval=50,
        checkpoint_interval=200,
        num_eval_games=20
    )
    
    # 开始训练
    # 注意：这里只训练100手牌作为示例，实际应用中应该训练更多
    trainer.train(num_hands=100)