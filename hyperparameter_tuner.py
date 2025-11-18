import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print(" tqdm not available. Install with: pip install tqdm")
    print("    Progress bars will be disabled.\n")

MARL_CODE_AVAILABLE = False
try:
    from marl_3d_hardware_validated import (
        NetworkConfig, 
        Network3D_Enhanced, 
        get_suburban_config,
        NodeType
    )
    MARL_CODE_AVAILABLE = True
    print(" MARL-3D code imported successfully!\n")
except ImportError as e:
    print(f" MARL-3D code not found: {e}")
    print("    Ensure 'marl_3d_hardware_validated.py' is in the same directory")
    print("    Exiting...\n")
    sys.exit(1)

# Defines the complete search space for all tunable parameters
@dataclass
class TuningSpace:
    mu_values: List[float] = None
    nu_values: List[float] = None
    
    learning_rate_values: List[float] = None
    discount_factor_values: List[float] = None
    
    local_weight_values: List[float] = None
    team_weight_values: List[float] = None
    fairness_weight_values: List[float] = None
    
    buffer_size_values: List[int] = None
    retention_slots_values: List[int] = None
    spatial_radius_values: List[float] = None
    
    grace_period_factor_values: List[float] = None
    
    beam_tolerance_values: List[float] = None
    
    p_ls_values: List[float] = None
    collision_reward_values: List[float] = None
    discovery_reward_values: List[float] = None
    known_neighbor_reward_values: List[float] = None
    nothing_reward_values: List[float] = None
    
    # Sets the default (full) search ranges for all parameters
    def __post_init__(self):
        if self.mu_values is None:
            self.mu_values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
        
        if self.nu_values is None:
            self.nu_values = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]
        
        if self.learning_rate_values is None:
            self.learning_rate_values = [0.005, 0.01, 0.015, 0.02, 0.03]
        
        if self.discount_factor_values is None:
            self.discount_factor_values = [0.90, 0.93, 0.95, 0.97]
        
        if self.local_weight_values is None:
            self.local_weight_values = [0.6, 0.65, 0.7, 0.75, 0.8]
        
        if self.team_weight_values is None:
            self.team_weight_values = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        if self.buffer_size_values is None:
            self.buffer_size_values = [300, 500, 700]
        
        if self.retention_slots_values is None:
            self.retention_slots_values = [150, 200, 250]
        
        if self.spatial_radius_values is None:
            self.spatial_radius_values = [120, 150, 180]
        
        if self.grace_period_factor_values is None:
            self.grace_period_factor_values = [2.5, 3.0, 3.5]
        
        if self.beam_tolerance_values is None:
            self.beam_tolerance_values = [1.0, 1.1, 1.15, 1.2]
        
        if self.p_ls_values is None:
            self.p_ls_values = [0.4, 0.5, 0.6]
        
        if self.collision_reward_values is None:
            self.collision_reward_values = [1.5, 2.0, 2.5, 3.0]
        
        if self.discovery_reward_values is None:
            self.discovery_reward_values = [0.8, 1.0, 1.2, 1.5]
        
        if self.known_neighbor_reward_values is None:
            self.known_neighbor_reward_values = [-0.5, -1.0, -1.5]
        
        if self.nothing_reward_values is None:
            self.nothing_reward_values = [-0.3, -0.5, -0.7, -1.0]
    
    # Returns a reduced search space for quick testing
    def get_quick_space(self):
        self.mu_values = [0.06, 0.08, 0.10]
        self.nu_values = [0.10, 0.12, 0.15]
        self.learning_rate_values = [0.01, 0.02]
        self.discount_factor_values = [0.95]
        self.local_weight_values = [0.7, 0.75]
        self.team_weight_values = [0.2, 0.25]
        self.buffer_size_values = [500]
        self.retention_slots_values = [200]
        self.spatial_radius_values = [150]
        self.grace_period_factor_values = [3.0]
        self.beam_tolerance_values = [1.15]
        self.p_ls_values = [0.5]
        
        self.collision_reward_values = [2.0, 2.5]
        self.discovery_reward_values = [1.0, 1.2]
        self.known_neighbor_reward_values = [-1.0, -1.5]
        self.nothing_reward_values = [-0.5, -0.7]
        return self

# A wrapper class to run simulations with different configurations
class SimulationRunner:
    # Initializes the runner with a base configuration
    def __init__(self, base_config: NetworkConfig):
        self.base_config = base_config
        self.results_cache = {}
    
    # Runs a single simulation trial with a given set of parameters
    def run_trial(self, config_dict: Dict, trial_seed: int = 42) -> Dict:
        config = copy.deepcopy(self.base_config)
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        np.random.seed(trial_seed)
        import random
        random.seed(trial_seed)
        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            network = Network3D_Enhanced(config)
            network.run_simulation()
            results = network.get_results()
            
            sys.stdout = old_stdout
            return results
            
        except Exception as e:
            sys.stdout = old_stdout
            print(f"\n Simulation error: {e}")
            return {
                'converged': False,
                'convergence_timeslot': None,
                'discovery_rate': 0.0,
                'avg_energy_per_node': 10.0,
                'collisions': 100,
                'ldr': 1.0,
                'jains_index': 0.0
            }
    
    # Runs multiple trials for a configuration and aggregates the results
    def run_multiple_trials(self, config_dict: Dict, num_trials: int = 3) -> Dict:
        trial_results = []
        
        for trial in range(num_trials):
            seed = 42 + trial
            result = self.run_trial(config_dict, seed)
            trial_results.append(result)
        
        converged_trials = [r for r in trial_results if r.get('converged', False)]
        
        return {
            'convergence_rate': len(converged_trials) / num_trials,
            'avg_conv_time': np.mean([r['convergence_timeslot'] for r in converged_trials]) if converged_trials else None,
            'std_conv_time': np.std([r['convergence_timeslot'] for r in converged_trials]) if converged_trials else None,
            'avg_discovery': np.mean([r['discovery_rate'] for r in trial_results]),
            'std_discovery': np.std([r['discovery_rate'] for r in trial_results]),
            'avg_energy': np.mean([r['avg_energy_per_node'] for r in trial_results]),
            'avg_collisions': np.mean([r['collisions'] for r in trial_results]),
            'avg_ldr': np.mean([r['ldr'] for r in trial_results]),
            'avg_fairness': np.mean([r['jains_index'] for r in trial_results]),
            'all_trials': trial_results
        }

# Manages the multi-phase hyperparameter tuning process
class MultiPhaseTuner:
    # Initializes the tuner with a search space and base config
    def __init__(self, search_space: TuningSpace, base_config: NetworkConfig):
        self.search_space = search_space
        self.base_config = base_config
        self.runner = SimulationRunner(base_config)
        
        self.phase_results = {
            'phase1_aerap': [],
            'phase2_rewards': [],
            'phase3_rl': [],
            'phase4_marl': [],
            'phase5_async3d': [],
            'phase6_reward_values': []
        }
        
        self.best_configs = {}
    
    # Computes an overall score for ranking configurations (lower is better)
    def compute_score(self, results: Dict, weights: Dict = None) -> float:
        if weights is None:
            weights = {
                'conv_time': 0.5,
                'discovery': 0.3,
                'energy': 0.1,
                'fairness': 0.1
            }
        
        if results['convergence_rate'] < 0.5:
            return 10000.0
        
        conv_time = results['avg_conv_time'] if results['avg_conv_time'] else 1000
        discovery = results['avg_discovery']
        energy = results['avg_energy']
        fairness = results['avg_fairness']
        
        score = (
            weights['conv_time'] * (conv_time / 600) +
            weights['discovery'] * (1.0 - discovery / 0.92) +
            weights['energy'] * (energy / 3.0) +
            weights['fairness'] * (1.0 - fairness / 0.75)
        )
        
        return score
    
    # Phase 1: Tunes AERAP parameters (mu, nu)
    def tune_aerap(self, num_trials: int = 3) -> Dict:
        print("\n" + "="*80)
        print("PHASE 1: TUNING AERAP PARAMETERS (mu, nu)")
        print("="*80)
        print(f"Search space: {len(self.search_space.mu_values)} × {len(self.search_space.nu_values)}")
        print(f"Total configs: {len(self.search_space.mu_values) * len(self.search_space.nu_values)}")
        print(f"Trials per config: {num_trials}")
        print(f"Total simulations: {len(self.search_space.mu_values) * len(self.search_space.nu_values) * num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        total = len(self.search_space.mu_values) * len(self.search_space.nu_values)
        
        if TQDM_AVAILABLE:
            iterator = tqdm(total=total, desc="AERAP Tuning")
        else:
            iterator = None
            counter = 0
        
        for mu in self.search_space.mu_values:
            for nu in self.search_space.nu_values:
                
                config_dict = {
                    'mu': mu,
                    'nu': nu
                }
                
                results = self.runner.run_multiple_trials(config_dict, num_trials)
                score = self.compute_score(results)
                
                result_entry = {
                    'mu': mu,
                    'nu': nu,
                    'score': score,
                    **results
                }
                
                self.phase_results['phase1_aerap'].append(result_entry)
                
                if score < best_score:
                    best_score = score
                    best_config = result_entry
                
                if iterator:
                    iterator.update(1)
                else:
                    counter += 1
                    print(f"    Progress: {counter}/{total} - Current best: mu={best_config['mu']:.3f}, nu={best_config['nu']:.3f}")
        
        if iterator:
            iterator.close()
        
        self.best_configs['aerap'] = best_config
        
        print(f"\nPHASE 1 COMPLETE!")
        print(f"    Best mu: {best_config['mu']:.3f}")
        print(f"    Best nu: {best_config['nu']:.3f}")
        if best_config['avg_conv_time']:
            print(f"    Avg Conv Time: {best_config['avg_conv_time']:.1f} slots")
        print(f"    Avg Discovery: {best_config['avg_discovery']:.2%}")
        print(f"    Conv Rate: {best_config['convergence_rate']:.0%}")
        
        return best_config
    
    # Phase 2: Tunes reward weights (local, team, fairness)
    def tune_reward_weights(self, num_trials: int = 2) -> Dict:
        print("\n" + "="*80)
        print("PHASE 2: TUNING REWARD WEIGHTS")
        print("="*80)
        
        best_aerap = self.best_configs['aerap']
        
        weight_combos = []
        for local in self.search_space.local_weight_values:
            for team in self.search_space.team_weight_values:
                fairness = 1.0 - local - team
                if 0.0 <= fairness <= 0.4:
                    weight_combos.append((local, team, fairness))
        
        print(f"Testing {len(weight_combos)} weight combinations")
        print(f"Trials per config: {num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        if TQDM_AVAILABLE:
            iterator = tqdm(weight_combos, desc="Reward Tuning")
        else:
            iterator = weight_combos
            counter = 0
        
        for local_w, team_w, fair_w in iterator:
            
            config_dict = {
                'mu': best_aerap['mu'],
                'nu': best_aerap['nu'],
                'reward_local_weight': local_w,
                'reward_team_weight': team_w,
                'reward_fairness_weight': fair_w
            }
            
            results = self.runner.run_multiple_trials(config_dict, num_trials)
            score = self.compute_score(results)
            
            result_entry = {
                'local_weight': local_w,
                'team_weight': team_w,
                'fairness_weight': fair_w,
                'score': score,
                **results
            }
            
            self.phase_results['phase2_rewards'].append(result_entry)
            
            if score < best_score:
                best_score = score
                best_config = result_entry
            
            if not TQDM_AVAILABLE:
                counter += 1
                if counter % 5 == 0:
                    print(f"    Progress: {counter}/{len(weight_combos)}")
        
        self.best_configs['rewards'] = best_config
        
        print(f"\nPHASE 2 COMPLETE!")
        print(f"    Best Local Weight: {best_config['local_weight']:.2f}")
        print(f"    Best Team Weight: {best_config['team_weight']:.2f}")
        print(f"    Best Fairness Weight: {best_config['fairness_weight']:.2f}")
        
        return best_config
    
    # Phase 3: Tunes RL parameters (learning_rate, discount_factor)
    def tune_rl_parameters(self, num_trials: int = 2) -> Dict:
        print("\n" + "="*80)
        print("PHASE 3: TUNING RL PARAMETERS")
        print("="*80)
        
        best_aerap = self.best_configs['aerap']
        best_rewards = self.best_configs['rewards']
        
        total = len(self.search_space.learning_rate_values) * len(self.search_space.discount_factor_values)
        print(f"Testing {total} RL parameter combinations")
        print(f"Trials per config: {num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        if TQDM_AVAILABLE:
            iterator = tqdm(total=total, desc="RL Tuning")
        else:
            iterator = None
            counter = 0
        
        for lr in self.search_space.learning_rate_values:
            for gamma in self.search_space.discount_factor_values:
                
                config_dict = {
                    'mu': best_aerap['mu'],
                    'nu': best_aerap['nu'],
                    'reward_local_weight': best_rewards['local_weight'],
                    'reward_team_weight': best_rewards['team_weight'],
                    'reward_fairness_weight': best_rewards['fairness_weight'],
                    'learning_rate': lr,
                    'discount_factor': gamma
                }
                
                results = self.runner.run_multiple_trials(config_dict, num_trials)
                score = self.compute_score(results)
                
                result_entry = {
                    'learning_rate': lr,
                    'discount_factor': gamma,
                    'score': score,
                    **results
                }
                
                self.phase_results['phase3_rl'].append(result_entry)
                
                if score < best_score:
                    best_score = score
                    best_config = result_entry
                
                if iterator:
                    iterator.update(1)
                else:
                    counter += 1
                    print(f"    Progress: {counter}/{total}")
        
        if iterator:
            iterator.close()
        
        self.best_configs['rl'] = best_config
        
        print(f"\nPHASE 3 COMPLETE!")
        print(f"    Best Learning Rate: {best_config['learning_rate']:.3f}")
        print(f"    Best Discount Factor: {best_config['discount_factor']:.2f}")
        
        return best_config
    
    # Phase 4: Tunes MARL parameters (buffer, retention, radius)
    def tune_marl_parameters(self, num_trials: int = 2) -> Dict:
        print("\n" + "="*80)
        print("PHASE 4: TUNING MARL PARAMETERS")
        print("="*80)
        
        prev_best = {
            'mu': self.best_configs['aerap']['mu'],
            'nu': self.best_configs['aerap']['nu'],
            'reward_local_weight': self.best_configs['rewards']['local_weight'],
            'reward_team_weight': self.best_configs['rewards']['team_weight'],
            'reward_fairness_weight': self.best_configs['rewards']['fairness_weight'],
            'learning_rate': self.best_configs['rl']['learning_rate'],
            'discount_factor': self.best_configs['rl']['discount_factor']
        }
        
        marl_combos = []
        for buf_size in self.search_space.buffer_size_values:
            for retention in self.search_space.retention_slots_values:
                for radius in self.search_space.spatial_radius_values:
                    marl_combos.append((buf_size, retention, radius))
        
        print(f"Testing {len(marl_combos)} MARL parameter combinations")
        print(f"Trials per config: {num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        if TQDM_AVAILABLE:
            iterator = tqdm(marl_combos, desc="MARL Tuning")
        else:
            iterator = marl_combos
            counter = 0
        
        for buf_size, retention, radius in iterator:
            
            config_dict = {
                **prev_best,
                'experience_buffer_size': buf_size,
                'experience_retention_slots': retention,
                'spatial_relevance_radius': radius
            }
            
            results = self.runner.run_multiple_trials(config_dict, num_trials)
            score = self.compute_score(results)
            
            result_entry = {
                'buffer_size': buf_size,
                'retention_slots': retention,
                'spatial_radius': radius,
                'score': score,
                **results
            }
            
            self.phase_results['phase4_marl'].append(result_entry)
            
            if score < best_score:
                best_score = score
                best_config = result_entry
            
            if not TQDM_AVAILABLE:
                counter += 1
                if counter % 3 == 0:
                    print(f"    Progress: {counter}/{len(marl_combos)}")
        
        self.best_configs['marl'] = best_config
        
        print(f"\nPHASE 4 COMPLETE!")
        print(f"    Best Buffer Size: {best_config['buffer_size']}")
        print(f"    Best Retention Slots: {best_config['retention_slots']}")
        print(f"    Best Spatial Radius: {best_config['spatial_radius']:.1f}m")
        
        return best_config
    
    # Phase 5: Tunes Async/3D parameters (grace, beam_tol, p_ls)
    def tune_async3d_parameters(self, num_trials: int = 2) -> Dict:
        print("\n" + "="*80)
        print("PHASE 5: TUNING ASYNC/3D PARAMETERS")
        print("="*80)
        
        prev_best = {
            'mu': self.best_configs['aerap']['mu'],
            'nu': self.best_configs['aerap']['nu'],
            'reward_local_weight': self.best_configs['rewards']['local_weight'],
            'reward_team_weight': self.best_configs['rewards']['team_weight'],
            'reward_fairness_weight': self.best_configs['rewards']['fairness_weight'],
            'learning_rate': self.best_configs['rl']['learning_rate'],
            'discount_factor': self.best_configs['rl']['discount_factor'],
            'experience_buffer_size': self.best_configs['marl']['buffer_size'],
            'experience_retention_slots': self.best_configs['marl']['retention_slots'],
            'spatial_relevance_radius': self.best_configs['marl']['spatial_radius']
        }
        
        async3d_combos = []
        for grace in self.search_space.grace_period_factor_values:
            for beam_tol in self.search_space.beam_tolerance_values:
                for p_ls in self.search_space.p_ls_values:
                    async3d_combos.append((grace, beam_tol, p_ls))
        
        print(f"Testing {len(async3d_combos)} Async/3D parameter combinations")
        print(f"Trials per config: {num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        if TQDM_AVAILABLE:
            iterator = tqdm(async3d_combos, desc="Async/3D Tuning")
        else:
            iterator = async3d_combos
            counter = 0
        
        for grace, beam_tol, p_ls in iterator:
            
            config_dict = {
                **prev_best,
                'grace_period_factor': grace,
                'beam_alignment_tolerance': beam_tol,
                'p_ls': p_ls
            }
            
            results = self.runner.run_multiple_trials(config_dict, num_trials)
            score = self.compute_score(results)
            
            result_entry = {
                'grace_period_factor': grace,
                'beam_tolerance': beam_tol,
                'p_ls': p_ls,
                'score': score,
                **results
            }
            
            self.phase_results['phase5_async3d'].append(result_entry)
            
            if score < best_score:
                best_score = score
                best_config = result_entry
            
            if not TQDM_AVAILABLE:
                counter += 1
                if counter % 5 == 0:
                    print(f"    Progress: {counter}/{len(async3d_combos)}")
        
        self.best_configs['async3d'] = best_config
        
        print(f"\nPHASE 5 COMPLETE!")
        print(f"    Best Grace Period Factor: {best_config['grace_period_factor']:.1f}")
        print(f"    Best Beam Tolerance: {best_config['beam_tolerance']:.2f}")
        print(f"    Best p_ls: {best_config['p_ls']:.1f}")
        
        return best_config
    
    # Phase 6: Tunes the specific reward magnitudes for observations
    def tune_reward_values(self, num_trials: int = 2) -> Dict:
        print("\n" + "="*80)
        print("PHASE 6: TUNING REWARD VALUES")
        print("="*80)
        print("This tunes the actual reward magnitudes for:")
        print("    - Collision detection")
        print("    - New neighbor discovery")
        print("    - Known neighbor rediscovery")
        print("    - Nothing heard (idle)")
        
        prev_best = {
            'mu': self.best_configs['aerap']['mu'],
            'nu': self.best_configs['aerap']['nu'],
            'reward_local_weight': self.best_configs['rewards']['local_weight'],
            'reward_team_weight': self.best_configs['rewards']['team_weight'],
            'reward_fairness_weight': self.best_configs['rewards']['fairness_weight'],
            'learning_rate': self.best_configs['rl']['learning_rate'],
            'discount_factor': self.best_configs['rl']['discount_factor'],
            'experience_buffer_size': self.best_configs['marl']['buffer_size'],
            'experience_retention_slots': self.best_configs['marl']['retention_slots'],
            'spatial_relevance_radius': self.best_configs['marl']['spatial_radius'],
            'grace_period_factor': self.best_configs['async3d']['grace_period_factor'],
            'beam_alignment_tolerance': self.best_configs['async3d']['beam_tolerance'],
            'p_ls': self.best_configs['async3d']['p_ls']
        }
        
        reward_combos = []
        for collision_r in self.search_space.collision_reward_values:
            for discovery_r in self.search_space.discovery_reward_values:
                for known_r in self.search_space.known_neighbor_reward_values:
                    for nothing_r in self.search_space.nothing_reward_values:
                        if collision_r > discovery_r:
                            reward_combos.append((collision_r, discovery_r, known_r, nothing_r))
        
        print(f"\nTesting {len(reward_combos)} reward value combinations")
        print(f"Trials per config: {num_trials}\n")
        
        best_score = float('inf')
        best_config = None
        
        if TQDM_AVAILABLE:
            iterator = tqdm(reward_combos, desc="Reward Value Tuning")
        else:
            iterator = reward_combos
            counter = 0
        
        for collision_r, discovery_r, known_r, nothing_r in iterator:
            
            config_dict = {
                **prev_best,
                'reward_collision': collision_r,
                'reward_discovery': discovery_r,
                'reward_known_neighbor': known_r,
                'reward_nothing': nothing_r
            }
            
            results = self.runner.run_multiple_trials(config_dict, num_trials)
            score = self.compute_score(results)
            
            result_entry = {
                'collision_reward': collision_r,
                'discovery_reward': discovery_r,
                'known_neighbor_reward': known_r,
                'nothing_reward': nothing_r,
                'score': score,
                **results
            }
            
            self.phase_results['phase6_reward_values'].append(result_entry)
            
            if score < best_score:
                best_score = score
                best_config = result_entry
            
            if not TQDM_AVAILABLE:
                counter += 1
                if counter % 10 == 0:
                    print(f"    Progress: {counter}/{len(reward_combos)}")
        
        self.best_configs['reward_values'] = best_config
        
        print(f"\nPHASE 6 COMPLETE!")
        print(f"    Best Collision Reward: {best_config['collision_reward']:.1f}")
        print(f"    Best Discovery Reward: {best_config['discovery_reward']:.1f}")
        print(f"    Best Known Neighbor Penalty: {best_config['known_neighbor_reward']:.1f}")
        print(f"    Best Nothing Penalty: {best_config['nothing_reward']:.1f}")
        
        return best_config
    
    # Runs all 6 tuning phases sequentially
    def run_complete_tuning(self, trials_per_phase: List[int] = None):
        if trials_per_phase is None:
            trials_per_phase = [3, 2, 2, 2, 2, 2]
        
        print("\n" + "="*80)
        print("MARL-3D: COMPLETE 6-PHASE HYPERPARAMETER TUNING")
        print("="*80)
        print("\nThis will sequentially optimize:")
        print("    Phase 1: AERAP parameters (mu, nu)")
        print("    Phase 2: Reward weights (local, team, fairness)")
        print("    Phase 3: RL parameters (learning_rate, discount_factor)")
        print("    Phase 4: MARL parameters (buffer, retention, radius)")
        print("    Phase 5: Async/3D parameters (grace, beam_tol, p_ls)")
        print("    Phase 6: Reward values (collision, discovery, penalties)  <- NEW!")
        print("\nEstimated time: 30-50 minutes\n")
        
        input("Press ENTER to start complete tuning...")
        
        start_time = datetime.now()
        
        self.tune_aerap(num_trials=trials_per_phase[0])
        self.tune_reward_weights(num_trials=trials_per_phase[1])
        self.tune_rl_parameters(num_trials=trials_per_phase[2])
        self.tune_marl_parameters(num_trials=trials_per_phase[3])
        self.tune_async3d_parameters(num_trials=trials_per_phase[4])
        
        self.tune_reward_values(num_trials=trials_per_phase[5])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("ALL 6 PHASES COMPLETE!")
        print("="*80)
        print(f"\nTotal tuning time: {duration:.1f} minutes")
        
        return self.get_optimal_config()
    
    # Compiles the best parameters from all phases into a single dictionary
    def get_optimal_config(self) -> Dict:
        optimal = {
            'mu': self.best_configs['aerap']['mu'],
            'nu': self.best_configs['aerap']['nu'],
            
            'reward_local_weight': self.best_configs['rewards']['local_weight'],
            'reward_team_weight': self.best_configs['rewards']['team_weight'],
            'reward_fairness_weight': self.best_configs['rewards']['fairness_weight'],
            
            'learning_rate': self.best_configs['rl']['learning_rate'],
            'discount_factor': self.best_configs['rl']['discount_factor'],
            
            'experience_buffer_size': self.best_configs['marl']['buffer_size'],
            'experience_retention_slots': self.best_configs['marl']['retention_slots'],
            'spatial_relevance_radius': self.best_configs['marl']['spatial_radius'],
            
            'grace_period_factor': self.best_configs['async3d']['grace_period_factor'],
            'beam_alignment_tolerance': self.best_configs['async3d']['beam_tolerance'],
            'p_ls': self.best_configs['async3d']['p_ls'],
            
            'reward_collision': self.best_configs['reward_values']['collision_reward'],
            'reward_discovery': self.best_configs['reward_values']['discovery_reward'],
            'reward_known_neighbor': self.best_configs['reward_values']['known_neighbor_reward'],
            'reward_nothing': self.best_configs['reward_values']['nothing_reward'],
            
            'performance': {
                'avg_conv_time': self.best_configs['aerap'].get('avg_conv_time'),
                'avg_discovery': self.best_configs['aerap']['avg_discovery'],
                'convergence_rate': self.best_configs['aerap']['convergence_rate'],
                'avg_energy': self.best_configs['aerap']['avg_energy'],
                'avg_fairness': self.best_configs['aerap']['avg_fairness']
            }
        }
        
        return optimal

# Provides static methods for plotting tuning results
class TuningVisualizer:
    
    # Plots 2x2 heatmaps for AERAP (mu vs nu) tuning results
    @staticmethod
    def plot_aerap_heatmaps(results: List[Dict], save_path: str = 'aerap_tuning.png'):
        if not results:
            print(" No AERAP results to plot")
            return None
        
        mu_values = sorted(list(set(r['mu'] for r in results)))
        nu_values = sorted(list(set(r['nu'] for r in results)))
        
        conv_matrix = np.full((len(nu_values), len(mu_values)), np.nan)
        disc_matrix = np.full((len(nu_values), len(mu_values)), np.nan)
        energy_matrix = np.full((len(nu_values), len(mu_values)), np.nan)
        fair_matrix = np.full((len(nu_values), len(mu_values)), np.nan)
        
        for r in results:
            i = nu_values.index(r['nu'])
            j = mu_values.index(r['mu'])
            conv_matrix[i, j] = r['avg_conv_time'] if r['avg_conv_time'] else 1000
            disc_matrix[i, j] = r['avg_discovery']
            energy_matrix[i, j] = r['avg_energy']
            fair_matrix[i, j] = r['avg_fairness']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        ax1 = axes[0, 0]
        im1 = ax1.imshow(conv_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax1.set_xticks(range(len(mu_values)))
        ax1.set_yticks(range(len(nu_values)))
        ax1.set_xticklabels([f'{m:.2f}' for m in mu_values])
        ax1.set_yticklabels([f'{n:.2f}' for n in nu_values])
        ax1.set_xlabel('μ (Reward Learning Rate)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ν (Penalty Learning Rate)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Convergence Time (slots)', fontsize=13, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        
        for i in range(len(nu_values)):
            for j in range(len(mu_values)):
                if not np.isnan(conv_matrix[i, j]) and conv_matrix[i, j] < 1000:
                    val = int(conv_matrix[i, j])
                    color = 'white' if conv_matrix[i, j] < np.nanmean(conv_matrix[conv_matrix < 1000]) else 'black'
                    ax1.text(j, i, str(val), ha='center', va='center', 
                            color=color, fontsize=9, fontweight='bold')
        
        ax2 = axes[0, 1]
        im2 = ax2.imshow(disc_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax2.set_xticks(range(len(mu_values)))
        ax2.set_yticks(range(len(nu_values)))
        ax2.set_xticklabels([f'{m:.2f}' for m in mu_values])
        ax2.set_yticklabels([f'{n:.2f}' for n in nu_values])
        ax2.set_xlabel('μ (Reward Learning Rate)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ν (Penalty Learning Rate)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Discovery Rate', fontsize=13, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2)
        
        for i in range(len(nu_values)):
            for j in range(len(mu_values)):
                if not np.isnan(disc_matrix[i, j]):
                    val = disc_matrix[i, j]
                    color = 'white' if val > np.nanmean(disc_matrix) else 'black'
                    ax2.text(j, i, f'{val:.2%}', ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')
        
        ax3 = axes[1, 0]
        im3 = ax3.imshow(energy_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax3.set_xticks(range(len(mu_values)))
        ax3.set_yticks(range(len(nu_values)))
        ax3.set_xticklabels([f'{m:.2f}' for m in mu_values])
        ax3.set_yticklabels([f'{n:.2f}' for n in nu_values])
        ax3.set_xlabel('μ (Reward Learning Rate)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('ν (Penalty Learning Rate)', fontsize=12, fontweight='bold')
        ax3.set_title('(c) Energy per Node (J)', fontsize=13, fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=ax3)
        
        for i in range(len(nu_values)):
            for j in range(len(mu_values)):
                if not np.isnan(energy_matrix[i, j]):
                    val = energy_matrix[i, j]
                    color = 'white' if val < np.nanmean(energy_matrix) else 'black'
                    ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')
        
        ax4 = axes[1, 1]
        im4 = ax4.imshow(fair_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax4.set_xticks(range(len(mu_values)))
        ax4.set_yticks(range(len(nu_values)))
        ax4.set_xticklabels([f'{m:.2f}' for m in mu_values])
        ax4.set_yticklabels([f'{n:.2f}' for n in nu_values])
        ax4.set_xlabel('μ (Reward Learning Rate)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('ν (Penalty Learning Rate)', fontsize=12, fontweight='bold')
        ax4.set_title("(d) Jain's Fairness Index", fontsize=13, fontweight='bold')
        cbar4 = plt.colorbar(im4, ax=ax4)
        
        for i in range(len(nu_values)):
            for j in range(len(mu_values)):
                if not np.isnan(fair_matrix[i, j]):
                    val = fair_matrix[i, j]
                    color = 'white' if val > np.nanmean(fair_matrix) else 'black'
                    ax4.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=9, fontweight='bold')
        
        plt.suptitle('Phase 1: AERAP Parameter Sensitivity Analysis', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAERAP heatmaps saved: {save_path}")
        
        return fig
    
    # Plots scatter plots for reward weight tuning
    @staticmethod
    def plot_reward_weights(results: List[Dict], save_path: str = 'reward_weights.png'):
        if not results:
            print(" No reward weight results to plot")
            return None
        
        local_weights = [r['local_weight'] for r in results]
        team_weights = [r['team_weight'] for r in results]
        conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for r in results]
        discovery_rates = [r['avg_discovery'] for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        scatter1 = ax1.scatter(local_weights, team_weights, c=conv_times, 
                                s=150, cmap='RdYlGn_r', alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Local Weight', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Team Weight', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Convergence Time vs Reward Weights', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Conv Time (slots)')
        
        ax2 = axes[1]
        scatter2 = ax2.scatter(local_weights, team_weights, c=discovery_rates,
                                s=150, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Local Weight', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Team Weight', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Discovery Rate vs Reward Weights', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='Discovery Rate')
        
        plt.suptitle('Phase 2: Reward Weight Optimization', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward weight plots saved: {save_path}")
        
        return fig
    
    # Plots heatmaps for RL parameter (LR vs gamma) tuning
    @staticmethod
    def plot_rl_parameters(results: List[Dict], save_path: str = 'rl_parameters.png'):
        if not results:
            print(" No RL parameter results to plot")
            return None
        
        lr_values = sorted(list(set(r['learning_rate'] for r in results)))
        gamma_values = sorted(list(set(r['discount_factor'] for r in results)))
        
        conv_matrix = np.full((len(gamma_values), len(lr_values)), np.nan)
        disc_matrix = np.full((len(gamma_values), len(lr_values)), np.nan)
        
        for r in results:
            i = gamma_values.index(r['discount_factor'])
            j = lr_values.index(r['learning_rate'])
            conv_matrix[i, j] = r['avg_conv_time'] if r['avg_conv_time'] else 1000
            disc_matrix[i, j] = r['avg_discovery']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        im1 = ax1.imshow(conv_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        ax1.set_xticks(range(len(lr_values)))
        ax1.set_yticks(range(len(gamma_values)))
        ax1.set_xticklabels([f'{lr:.3f}' for lr in lr_values])
        ax1.set_yticklabels([f'{g:.2f}' for g in gamma_values])
        ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Discount Factor (γ)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Convergence Time', fontsize=13, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Conv Time (slots)')
        
        for i in range(len(gamma_values)):
            for j in range(len(lr_values)):
                if not np.isnan(conv_matrix[i, j]) and conv_matrix[i, j] < 1000:
                    val = int(conv_matrix[i, j])
                    color = 'white' if conv_matrix[i, j] < np.nanmean(conv_matrix[conv_matrix < 1000]) else 'black'
                    ax1.text(j, i, str(val), ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')
        
        ax2 = axes[1]
        im2 = ax2.imshow(disc_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax2.set_xticks(range(len(lr_values)))
        ax2.set_yticks(range(len(gamma_values)))
        ax2.set_xticklabels([f'{lr:.3f}' for lr in lr_values])
        ax2.set_yticklabels([f'{g:.2f}' for g in gamma_values])
        ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Discount Factor (γ)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Discovery Rate', fontsize=13, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Discovery Rate')
        
        for i in range(len(gamma_values)):
            for j in range(len(lr_values)):
                if not np.isnan(disc_matrix[i, j]):
                    val = disc_matrix[i, j]
                    color = 'white' if val > np.nanmean(disc_matrix) else 'black'
                    ax2.text(j, i, f'{val:.2%}', ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')
        
        plt.suptitle('Phase 3: RL Parameter Optimization',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RL parameter plots saved: {save_path}")
        
        return fig
    
    # Creates a summary dashboard plot of all tuning phases
    @staticmethod
    def plot_phase_summary(tuner: MultiPhaseTuner, save_path: str = 'tuning_summary.png'):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :2])
        aerap_results = tuner.phase_results['phase1_aerap']
        if aerap_results:
            scores = [r['score'] for r in aerap_results if r['score'] < 10000]
            conv_times = [r['avg_conv_time'] for r in aerap_results if r['avg_conv_time']]
            if scores:
                ax1.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                ax1.set_xlabel('Score', fontweight='bold')
                ax1.set_ylabel('Frequency', fontweight='bold')
                ax1.set_title('Phase 1: AERAP Score Distribution', fontweight='bold')
                ax1.axvline(min(scores), color='red', linestyle='--', label=f'Best: {min(scores):.1f}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2])
        reward_results = tuner.phase_results['phase2_rewards']
        if reward_results:
            best_idx = np.argmin([r['score'] for r in reward_results])
            best = reward_results[best_idx]
            weights = ['Local', 'Team', 'Fair']
            values = [best['local_weight'], best['team_weight'], best['fairness_weight']]
            ax2.pie(values, labels=weights, autopct='%1.1f%%', startangle=90,
                    colors=['#ff9999', '#66b3ff', '#99ff99'])
            ax2.set_title('Phase 2: Best Reward Weights', fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, 0])
        rl_results = tuner.phase_results['phase3_rl']
        if rl_results:
            lrs = [r['learning_rate'] for r in rl_results]
            scores = [r['score'] for r in rl_results if r['score'] < 10000]
            if scores:
                ax3.scatter(lrs, scores, alpha=0.6, s=100, c=scores, cmap='RdYlGn_r')
                ax3.set_xlabel('Learning Rate', fontweight='bold')
                ax3.set_ylabel('Score', fontweight='bold')
                ax3.set_title('Phase 3: RL Learning Rate', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        marl_results = tuner.phase_results['phase4_marl']
        if marl_results:
            buffer_sizes = [r['buffer_size'] for r in marl_results]
            discovery_rates = [r['avg_discovery'] for r in marl_results]
            ax4.scatter(buffer_sizes, discovery_rates, alpha=0.6, s=100, 
                        c=discovery_rates, cmap='RdYlGn')
            ax4.set_xlabel('Buffer Size', fontweight='bold')
            ax4.set_ylabel('Discovery Rate', fontweight='bold')
            ax4.set_title('Phase 4: MARL Buffer Size', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 2])
        if marl_results:
            radii = [r['spatial_radius'] for r in marl_results]
            conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for r in marl_results]
            ax5.scatter(radii, conv_times, alpha=0.6, s=100, c=conv_times, cmap='RdYlGn_r')
            ax5.set_xlabel('Spatial Radius (m)', fontweight='bold')
            ax5.set_ylabel('Conv Time (slots)', fontweight='bold')
            ax5.set_title('Phase 4: Spatial Relevance', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[2, 0])
        async3d_results = tuner.phase_results['phase5_async3d']
        if async3d_results:
            grace_factors = [r['grace_period_factor'] for r in async3d_results]
            discovery_rates = [r['avg_discovery'] for r in async3d_results]
            ax6.scatter(grace_factors, discovery_rates, alpha=0.6, s=100,
                        c=discovery_rates, cmap='RdYlGn')
            ax6.set_xlabel('Grace Period Factor', fontweight='bold')
            ax6.set_ylabel('Discovery Rate', fontweight='bold')
            ax6.set_title('Phase 5: Grace Period', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[2, 1])
        if async3d_results:
            beam_tols = [r['beam_tolerance'] for r in async3d_results]
            conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for r in async3d_results]
            ax7.scatter(beam_tols, conv_times, alpha=0.6, s=100, c=conv_times, cmap='RdYlGn_r')
            ax7.set_xlabel('Beam Tolerance', fontweight='bold')
            ax7.set_ylabel('Conv Time (slots)', fontweight='bold')
            ax7.set_title('Phase 5: Beam Alignment', fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        summary_text = "OPTIMAL CONFIGURATION\n\n"
        if tuner.best_configs:
            if 'aerap' in tuner.best_configs:
                summary_text += f"μ: {tuner.best_configs['aerap']['mu']:.3f}\n"
                summary_text += f"ν: {tuner.best_configs['aerap']['nu']:.3f}\n\n"
            if 'rewards' in tuner.best_configs:
                summary_text += f"Local: {tuner.best_configs['rewards']['local_weight']:.2f}\n"
                summary_text += f"Team: {tuner.best_configs['rewards']['team_weight']:.2f}\n\n"
            if 'rl' in tuner.best_configs:
                summary_text += f"LR: {tuner.best_configs['rl']['learning_rate']:.3f}\n"
                summary_text += f"γ: {tuner.best_configs['rl']['discount_factor']:.2f}\n\n"
            if 'aerap' in tuner.best_configs and tuner.best_configs['aerap']['avg_conv_time']:
                summary_text += f"Conv: {tuner.best_configs['aerap']['avg_conv_time']:.0f} slots\n"
                summary_text += f"Disc: {tuner.best_configs['aerap']['avg_discovery']:.1%}"
        
        ax8.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle('MARL-3D: 5-Phase Hyperparameter Tuning Summary',
                    fontsize=16, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Complete tuning summary saved: {save_path}")
        
        return fig
    
    # Creates bar charts comparing baseline vs. optimized results
    @staticmethod
    def plot_baseline_vs_optimized(baseline_config: Dict, optimal_config: Dict, 
                                    baseline_results: Dict, optimal_results: Dict,
                                    save_path: str = 'comparison.png'):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        metrics = [
            ('Convergence Time\n(slots)', 
            baseline_results.get('avg_conv_time', 800),
            optimal_results.get('avg_conv_time', 650)),
            ('Discovery Rate\n(%)',
            baseline_results.get('avg_discovery', 0.85) * 100,
            optimal_results.get('avg_discovery', 0.92) * 100),
            ('Energy per Node\n(J)',
            baseline_results.get('avg_energy', 3.5),
            optimal_results.get('avg_energy', 3.2)),
            ('Collisions',
            baseline_results.get('avg_collisions', 20),
            optimal_results.get('avg_collisions', 15)),
            ('LDR',
            baseline_results.get('avg_ldr', 0.65),
            optimal_results.get('avg_ldr', 0.60)),
            ("Jain's Fairness",
            baseline_results.get('avg_fairness', 0.68),
            optimal_results.get('avg_fairness', 0.72))
        ]
        
        for idx, (ax, (metric, base_val, opt_val)) in enumerate(zip(axes.flat, metrics)):
            x = [0, 1]
            y = [base_val, opt_val]
            
            colors = ['coral', 'lightgreen']
            bars = ax.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(['Baseline', 'Optimized'], fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, y):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(y)*0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            if 'Time' in metric or 'Collisions' in metric or 'LDR' in metric:
                improvement = ((base_val - opt_val) / base_val) * 100
            else:
                improvement = ((opt_val - base_val) / base_val) * 100
            
            color = 'green' if improvement > 0 else 'red'
            ax.text(0.5, max(y)*1.15, f'{improvement:+.1f}%',
                    ha='center', fontsize=13, fontweight='bold', color=color)
        
        plt.suptitle('Baseline vs Optimized Configuration Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
        
        return fig
    
    # Plots scatter plots for the Phase 6 reward value tuning
    @staticmethod
    def plot_reward_values(results: List[Dict], save_path: str = 'reward_values_tuning.png'):
        if not results:
            print(" No reward value results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        collision_rewards = [r['collision_reward'] for r in results]
        discovery_rewards = [r['discovery_reward'] for r in results]
        conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for r in results]
        discovery_rates = [r['avg_discovery'] for r in results]
        
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(collision_rewards, conv_times, c=discovery_rates,
                                s=100, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Collision Reward', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Convergence Time (slots)', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Collision Reward Impact', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Discovery Rate')
        
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(discovery_rewards, discovery_rates, c=conv_times,
                                s=100, cmap='RdYlGn_r', alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Discovery Reward', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Discovery Rate', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Discovery Reward Impact', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Conv Time')
        
        ax3 = axes[1, 0]
        reward_ratios = [r['collision_reward'] / r['discovery_reward'] for r in results]
        ax3.scatter(reward_ratios, conv_times, c=discovery_rates,
                    s=100, cmap='RdYlGn', alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Collision/Discovery Ratio', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Convergence Time (slots)', fontsize=12, fontweight='bold')
        ax3.set_title('(c) Reward Ratio Impact', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        best_idx = np.argmin([r['score'] for r in results])
        best = results[best_idx]
        
        summary = f"BEST REWARD VALUES\n\n"
        summary += f"Collision: {best['collision_reward']:.1f}\n"
        summary += f"Discovery: {best['discovery_reward']:.1f}\n"
        summary += f"Known Neighbor: {best['known_neighbor_reward']:.1f}\n"
        summary += f"Nothing: {best['nothing_reward']:.1f}\n\n"
        summary += f"Performance:\n"
        if best['avg_conv_time']:
            summary += f"Conv Time: {best['avg_conv_time']:.0f}\n"
        summary += f"Discovery: {best['avg_discovery']:.1%}\n"
        summary += f"Conv Rate: {best['convergence_rate']:.0%}"
        
        ax4.text(0.5, 0.5, summary, fontsize=11, family='monospace',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle('Phase 6: Reward Value Optimization',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward value plots saved: {save_path}")
        
        if tuner.phase_results['phase6_reward_values']:
            vis.plot_reward_values(tuner.phase_results['phase6_reward_values'], 'reward_values_full.png')
        
        return fig

# Saves all tuning results and the optimal config to a JSON file
def save_results(tuner: MultiPhaseTuner, optimal_config: Dict, filename: str = 'tuning_results.json'):
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimal_config': optimal_config,
        'phase_results': {}
    }
    
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    for phase, data in tuner.phase_results.items():
        if data:
            results['phase_results'][phase] = []
            for entry in data:
                clean_entry = {k: v for k, v in entry.items() if k != 'all_trials'}
                results['phase_results'][phase].append(convert_types(clean_entry))
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete results saved: {filename}")

# Generates a LaTeX table of the optimal parameters
def generate_latex_table(optimal_config: Dict, filename: str = 'optimal_params.tex'):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Optimized Hyperparameters for MARL-3D}
\label{tab:optimal_hyperparameters}
\begin{tabular}{lll}
\hline
\textbf{Category} & \textbf{Parameter} & \textbf{Optimal Value} \\
\hline
"""
    
    latex += r"AERAP & $\mu$ (Reward rate) & " + f"{optimal_config['mu']:.3f} \\\\\n"
    latex += r"      & $\nu$ (Penalty rate) & " + f"{optimal_config['nu']:.3f} \\\\\n"
    latex += r"\hline" + "\n"
    
    latex += r"Reward & Local weight & " + f"{optimal_config['reward_local_weight']:.2f} \\\\\n"
    latex += r"Weights & Team weight & " + f"{optimal_config['reward_team_weight']:.2f} \\\\\n"
    latex += r"        & Fairness weight & " + f"{optimal_config['reward_fairness_weight']:.2f} \\\\\n"
    latex += r"\hline" + "\n"
    
    latex += r"RL & Learning rate ($\alpha$) & " + f"{optimal_config['learning_rate']:.3f} \\\\\n"
    latex += r"   & Discount factor ($\gamma$) & " + f"{optimal_config['discount_factor']:.2f} \\\\\n"
    latex += r"\hline" + "\n"
    
    latex += r"MARL & Buffer size & " + f"{optimal_config['experience_buffer_size']} \\\\\n"
    latex += r"      & Retention slots & " + f"{optimal_config['experience_retention_slots']} \\\\\n"
    latex += r"      & Spatial radius (m) & " + f"{optimal_config['spatial_relevance_radius']:.1f} \\\\\n"
    latex += r"\hline" + "\n"
    
    latex += r"Async/3D & Grace period factor & " + f"{optimal_config['grace_period_factor']:.1f} \\\\\n"
    latex += r"          & Beam tolerance & " + f"{optimal_config['beam_alignment_tolerance']:.2f} \\\\\n"
    latex += r"          & $p_{ls}$ & " + f"{optimal_config['p_ls']:.1f} \\\\\n"
    latex += r"\hline" + "\n"
    
    latex += r"""\end{tabular}
\end{table}
"""
    
    with open(filename, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved: {filename}")

# Prints instructions on how to use the optimal configuration
def print_usage_instructions(optimal_config: Dict):
    print("\n" + "="*80)
    print("HOW TO USE THESE OPTIMIZED PARAMETERS")
    print("="*80)
    
    print("\n1.  Update your NetworkConfig in marl_3d_hardware_validated.py:\n")
    
    print("config = NetworkConfig()")
    print(f"config.mu = {optimal_config['mu']:.3f}")
    print(f"config.nu = {optimal_config['nu']:.3f}")
    print(f"config.learning_rate = {optimal_config['learning_rate']:.3f}")
    print(f"config.discount_factor = {optimal_config['discount_factor']:.2f}")
    print(f"config.reward_local_weight = {optimal_config['reward_local_weight']:.2f}")
    print(f"config.reward_team_weight = {optimal_config['reward_team_weight']:.2f}")
    print(f"config.reward_fairness_weight = {optimal_config['reward_fairness_weight']:.2f}")
    print(f"config.experience_buffer_size = {optimal_config['experience_buffer_size']}")
    print(f"config.experience_retention_slots = {optimal_config['experience_retention_slots']}")
    print(f"config.spatial_relevance_radius = {optimal_config['spatial_relevance_radius']:.1f}")
    print(f"config.grace_period_factor = {optimal_config['grace_period_factor']:.1f}")
    print(f"config.beam_alignment_tolerance = {optimal_config['beam_alignment_tolerance']:.2f}")
    print(f"config.p_ls = {optimal_config['p_ls']:.1f}")
    
    print("\n2.  Run simulations with optimized parameters")
    print("\n3.  Report performance improvements in your paper:")
    
    perf = optimal_config.get('performance', {})
    if perf.get('avg_conv_time'):
        print(f"    - Convergence Time: {perf['avg_conv_time']:.1f} slots")
    print(f"    - Discovery Rate: {perf.get('avg_discovery', 0.0):.1%}")
    print(f"    - Convergence Rate: {perf.get('convergence_rate', 0.0):.0%}")
    
    print("\n4.  Include tuning methodology in paper:")
    print("    'Hyperparameters optimized via 5-phase grid search:")
    print("     Phase 1 (AERAP), Phase 2 (Rewards), Phase 3 (RL),")
    print("     Phase 4 (MARL), Phase 5 (Async/3D).'")

# The main execution function that shows the user menu and runs the selected tuning mode
def main():
    print("\n" + "="*80)
    print("MARL-3D COMPREHENSIVE HYPERPARAMETER TUNING FRAMEWORK v3.0")
    print("="*80)
    print("\nObjective: Find optimal parameters for fastest convergence")
    print("              with high discovery rate and low energy")
    print("\nTuning Strategy:")
    print("    Phase 1: AERAP parameters (mu, nu) - MOST CRITICAL")
    print("    Phase 2: Reward weights (local, team, fairness)")
    print("    Phase 3: RL parameters (learning_rate, discount_factor)")
    print("    Phase 4: MARL parameters (buffer, retention, radius)")
    print("    Phase 5: Async/3D parameters (grace, beam_tol, p_ls)")
    
    print("\nFixed Parameters (Hardware-Validated):")
    print("    - Communication radius: 100m (Ubiquiti NanoStation)")
    print("    - Mobility speeds: 3-8 m/s (DJI Matrice 600)")
    print("    - Beamwidth: 45° (from datasheet)")
    print("    - Clock drift: 2 ppm (Maxim DS3231)")
    
    print("\n" + "="*80)
    print("SELECT TUNING MODE:")
    print("="*80)
    print("\n1. Quick Mode (reduced search space, ~10-15 min)")
    print("    - 3×3 AERAP grid")
    print("    - Limited RL/MARL/Async combinations")
    print("    - Good for initial exploration")
    
    print("\n2. Full Mode (comprehensive search, ~30-45 min)")
    print("    - 7×7 AERAP grid")
    print("    - Complete RL/MARL/Async search")
    print("    - Recommended for final paper")
    
    print("\n3. AERAP Only (focus on critical params, ~15-20 min)")
    print("    - Only tune AERAP parameters")
    print("    - Use defaults for other parameters")
    print("    - Fast convergence optimization")
    
    print("\n4. Custom Mode (select specific phases)")
    print("    - Choose which phases to run")
    print("    - Flexible tuning strategy")
    
    print("\n5. Visualize Existing Results")
    print("    - Load and plot previous tuning results")
    
    print("\n6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    base_config = get_suburban_config()
    base_config.max_timeslots = 1000
    
    if choice == '1':
        print("\nQUICK MODE SELECTED")
        search_space = TuningSpace().get_quick_space()
        tuner = MultiPhaseTuner(search_space, base_config)
        
        optimal_config = tuner.run_complete_tuning(trials_per_phase=[2, 2, 2, 2, 2, 2])
        
        vis = TuningVisualizer()
        vis.plot_aerap_heatmaps(tuner.phase_results['phase1_aerap'], 'aerap_quick.png')
        vis.plot_phase_summary(tuner, 'summary_quick.png')
        
        save_results(tuner, optimal_config, 'tuning_results_quick.json')
        generate_latex_table(optimal_config, 'optimal_params_quick.tex')
        print_usage_instructions(optimal_config)
        
    elif choice == '2':
        print("\nFULL MODE SELECTED")
        print("    This will take 30-45 minutes...")
        confirm = input("    Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            search_space = TuningSpace()
            tuner = MultiPhaseTuner(search_space, base_config)
            
            optimal_config = tuner.run_complete_tuning(trials_per_phase=[3, 2, 2, 2, 2, 2])
            
            vis = TuningVisualizer()
            vis.plot_aerap_heatmaps(tuner.phase_results['phase1_aerap'], 'aerap_full.png')
            vis.plot_reward_weights(tuner.phase_results['phase2_rewards'], 'rewards_full.png')
            vis.plot_rl_parameters(tuner.phase_results['phase3_rl'], 'rl_full.png')
            vis.plot_phase_summary(tuner, 'summary_full.png')
            
            save_results(tuner, optimal_config, 'tuning_results_full.json')
            generate_latex_table(optimal_config, 'optimal_params_full.tex')
            print_usage_instructions(optimal_config)
            
            print("\n" + "="*80)
            print("BASELINE COMPARISON")
            print("="*80)
            print("\nRunning baseline configuration for comparison...")
            
            runner = SimulationRunner(base_config)
            baseline_results = runner.run_multiple_trials({}, num_trials=3)
            
            print("\nRunning optimized configuration...")
            optimal_results = runner.run_multiple_trials(
                {k: v for k, v in optimal_config.items() if k != 'performance'},
                num_trials=3
            )
            
            baseline_config = {}
            vis.plot_baseline_vs_optimized(
                baseline_config, optimal_config,
                baseline_results, optimal_results,
                'baseline_vs_optimized.png'
            )
            
        else:
            print("\nFull mode cancelled.")
            return
    
    elif choice == '3':
        print("\nAERAP-ONLY MODE SELECTED")
        print("    Focusing on critical AERAP parameters...")
        
        search_space = TuningSpace()
        tuner = MultiPhaseTuner(search_space, base_config)
        
        best_aerap = tuner.tune_aerap(num_trials=3)
        
        vis = TuningVisualizer()
        vis.plot_aerap_heatmaps(tuner.phase_results['phase1_aerap'], 'aerap_only.png')
        
        optimal_config = {
            'mu': best_aerap['mu'],
            'nu': best_aerap['nu'],
            'reward_local_weight': 0.7,
            'reward_team_weight': 0.2,
            'reward_fairness_weight': 0.1,
            'learning_rate': 0.01,
            'discount_factor': 0.95,
            'experience_buffer_size': 500,
            'experience_retention_slots': 200,
            'spatial_relevance_radius': 150.0,
            'grace_period_factor': 3.0,
            'beam_alignment_tolerance': 1.15,
            'p_ls': 0.5,
            'performance': {
                'avg_conv_time': best_aerap.get('avg_conv_time'),
                'avg_discovery': best_aerap['avg_discovery'],
                'convergence_rate': best_aerap['convergence_rate']
            }
        }
        
        save_results(tuner, optimal_config, 'tuning_results_aerap.json')
        print_usage_instructions(optimal_config)
    
    elif choice == '4':
        print("\nCUSTOM MODE SELECTED")
        print("\nSelect phases to run (separate with commas):")
        print("    1 = AERAP")
        print("    2 = Reward Weights")
        print("    3 = RL Parameters")
        print("    4 = MARL Parameters")
        print("    5 = Async/3D Parameters")
        
        phases_input = input("\nPhases to run (e.g., 1,2,3): ").strip()
        phases_to_run = [int(p) for p in phases_input.split(',') if p.strip().isdigit()]
        
        if not phases_to_run:
            print("No valid phases selected. Exiting.")
            return
        
        search_space = TuningSpace()
        tuner = MultiPhaseTuner(search_space, base_config)
        
        optimal_config = {
            'mu': 0.08,
            'nu': 0.10,
            'reward_local_weight': 0.7,
            'reward_team_weight': 0.2,
            'reward_fairness_weight': 0.1,
            'learning_rate': 0.01,
            'discount_factor': 0.95,
            'experience_buffer_size': 500,
            'experience_retention_slots': 200,
            'spatial_relevance_radius': 150.0,
            'grace_period_factor': 3.0,
            'beam_alignment_tolerance': 1.15,
            'p_ls': 0.5
        }
        
        if 1 in phases_to_run:
            best = tuner.tune_aerap(num_trials=3)
            optimal_config['mu'] = best['mu']
            optimal_config['nu'] = best['nu']
        
        if 2 in phases_to_run:
            best = tuner.tune_reward_weights(num_trials=2)
            optimal_config['reward_local_weight'] = best['local_weight']
            optimal_config['reward_team_weight'] = best['team_weight']
            optimal_config['reward_fairness_weight'] = best['fairness_weight']
        
        if 3 in phases_to_run:
            best = tuner.tune_rl_parameters(num_trials=2)
            optimal_config['learning_rate'] = best['learning_rate']
            optimal_config['discount_factor'] = best['discount_factor']
        
        if 4 in phases_to_run:
            best = tuner.tune_marl_parameters(num_trials=2)
            optimal_config['experience_buffer_size'] = best['buffer_size']
            optimal_config['experience_retention_slots'] = best['retention_slots']
            optimal_config['spatial_relevance_radius'] = best['spatial_radius']
        
        if 5 in phases_to_run:
            best = tuner.tune_async3d_parameters(num_trials=2)
            optimal_config['grace_period_factor'] = best['grace_period_factor']
            optimal_config['beam_alignment_tolerance'] = best['beam_tolerance']
            optimal_config['p_ls'] = best['p_ls']
        
        vis = TuningVisualizer()
        if 1 in phases_to_run:
            vis.plot_aerap_heatmaps(tuner.phase_results['phase1_aerap'], 'aerap_custom.png')
        if 2 in phases_to_run:
            vis.plot_reward_weights(tuner.phase_results['phase2_rewards'], 'rewards_custom.png')
        if 3 in phases_to_run:
            vis.plot_rl_parameters(tuner.phase_results['phase3_rl'], 'rl_custom.png')
        
        vis.plot_phase_summary(tuner, 'summary_custom.png')
        
        save_results(tuner, optimal_config, 'tuning_results_custom.json')
        print_usage_instructions(optimal_config)
    
    elif choice == '5':
        print("\nVISUALIZE EXISTING RESULTS")
        
        json_files = [f for f in os.listdir('.') if f.startswith('tuning_results') and f.endswith('.json')]
        
        if not json_files:
            print("\nNo tuning result files found.")
            print("    Run tuning first to generate results.")
            return
        
        print("\nAvailable result files:")
        for i, fname in enumerate(json_files, 1):
            print(f"    {i}. {fname}")
        
        file_choice = input("\nSelect file number: ").strip()
        
        try:
            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(json_files):
                filename = json_files[file_idx]
                
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                print(f"\nLoaded: {filename}")
                
                phase_results = data.get('phase_results', {})
                
                vis = TuningVisualizer()
                
                if 'phase1_aerap' in phase_results and phase_results['phase1_aerap']:
                    vis.plot_aerap_heatmaps(phase_results['phase1_aerap'], 'aerap_loaded.png')
                
                if 'phase2_rewards' in phase_results and phase_results['phase2_rewards']:
                    vis.plot_reward_weights(phase_results['phase2_rewards'], 'rewards_loaded.png')
                
                if 'phase3_rl' in phase_results and phase_results['phase3_rl']:
                    vis.plot_rl_parameters(phase_results['phase3_rl'], 'rl_loaded.png')
                
                print("\nVisualizations generated from loaded data")
                
                optimal = data.get('optimal_config', {})
                if optimal:
                    print("\n" + "="*80)
                    print("OPTIMAL CONFIGURATION FROM FILE")
                    print("="*80)
                    for key, value in optimal.items():
                        if key != 'performance':
                            print(f"    {key}: {value}")
                
                plt.show()
                
            else:
                print("Invalid file number")
        except (ValueError, IndexError):
            print("Invalid input")
    
    elif choice == '6':
        print("\nExiting. Run anytime to optimize parameters!")
        return
    
    else:
        print("\nInvalid choice. Exiting.")
        return
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    for fname in os.listdir('.'):
        if any(fname.startswith(prefix) for prefix in ['aerap_', 'rewards_', 'rl_', 'summary_', 
                                                        'comparison_', 'tuning_results', 'optimal_params']):
            print(f"    - {fname}")
    
    print("\nDisplaying plots...")
    plt.show()

# A utility function to test the sensitivity of a single parameter
def quick_sensitivity_analysis(param_name: str, values: List[float], 
                                base_config: NetworkConfig, num_trials: int = 3):
    print(f"\nSensitivity Analysis: {param_name}")
    print(f"    Testing {len(values)} values with {num_trials} trials each\n")
    
    runner = SimulationRunner(base_config)
    results = []
    
    for val in values:
        print(f"    Testing {param_name}={val:.3f}...", end=" ")
        
        config_dict = {param_name: val}
        trial_results = runner.run_multiple_trials(config_dict, num_trials)
        results.append(trial_results)
        
        if trial_results['avg_conv_time']:
            print(f"Conv: {trial_results['avg_conv_time']:.0f}, Disc: {trial_results['avg_discovery']:.1%}")
        else:
            print(f"No convergence, Disc: {trial_results['avg_discovery']:.1%}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for r in results]
    discoveries = [r['avg_discovery'] for r in results]
    energies = [r['avg_energy'] for r in results]
    
    axes[0].plot(values, conv_times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel(param_name, fontweight='bold')
    axes[0].set_ylabel('Convergence Time (slots)', fontweight='bold')
    axes[0].set_title('Convergence Time', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(values, discoveries, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel(param_name, fontweight='bold')
    axes[1].set_ylabel('Discovery Rate', fontweight='bold')
    axes[1].set_title('Discovery Rate', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(values, energies, 'o-', linewidth=2, markersize=8, color='orange')
    axes[2].set_xlabel(param_name, fontweight='bold')
    axes[2].set_ylabel('Energy per Node (J)', fontweight='bold')
    axes[2].set_title('Energy Consumption', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: sensitivity_{param_name}.png")
    plt.show()

# A utility function to compare the performance of several named configurations
def compare_configurations(configs: List[Tuple[str, Dict]], base_config: NetworkConfig, num_trials: int = 5):
    print(f"\nComparing {len(configs)} configurations")
    print(f"    {num_trials} trials per configuration\n")
    
    runner = SimulationRunner(base_config)
    all_results = []
    
    for name, config_dict in configs:
        print(f"    Running '{name}'...", end=" ")
        results = runner.run_multiple_trials(config_dict, num_trials)
        all_results.append((name, results))
        
        if results['avg_conv_time']:
            print(f"Conv: {results['avg_conv_time']:.0f}, Disc: {results['avg_discovery']:.1%}")
        else:
            print(f"No convergence, Disc: {results['avg_discovery']:.1%}")
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*88)
    print(f"\n{'Configuration':<20} {'Conv Time':<12} {'Discovery':<12} {'Energy':<12} {'Fairness':<12}")
    print("-" * 80)
    
    for name, results in all_results:
        conv_str = f"{results['avg_conv_time']:.0f}" if results['avg_conv_time'] else "N/A"
        print(f"{name:<20} {conv_str:<12} {results['avg_discovery']:<12.1%} "
            f"{results['avg_energy']:<12.2f} {results['avg_fairness']:<12.2f}")
    
    print("-" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [name for name, _ in all_results]
    conv_times = [r['avg_conv_time'] if r['avg_conv_time'] else 1000 for _, r in all_results]
    discoveries = [r['avg_discovery'] for _, r in all_results]
    energies = [r['avg_energy'] for _, r in all_results]
    fairness = [r['avg_fairness'] for _, r in all_results]
    
    x = range(len(names))
    
    axes[0, 0].bar(x, conv_times, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Convergence Time (slots)', fontweight='bold')
    axes[0, 0].set_title('Convergence Time', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(x, discoveries, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Discovery Rate', fontweight='bold')
    axes[0, 1].set_title('Discovery Rate', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar(x, energies, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Energy per Node (J)', fontweight='bold')
    axes[1, 0].set_title('Energy Consumption', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].bar(x, fairness, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].set_ylabel("Jain's Fairness Index", fontweight='bold')
    axes[1, 1].set_title('Fairness', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Configuration Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('configuration_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: configuration_comparison.png")
    plt.show()

# Standard Python entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        print("    Partial results may be saved.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error and ensure:")
        print("    1. marl_3d_hardware_validated.py is in the same directory")
        print("    2. All required packages are installed (numpy, matplotlib, tqdm)")
        print("    3. You have write permissions in the current directory")
    finally:
        print("\n" + "="*80)
        print("MARL-3D Hyperparameter Tuning Framework v3.0")
        print("Thank you for using the tuning framework!")
        print("="*80)


def load_and_compare_configs(baseline_json: str, optimized_json: str):
    
    with open(baseline_json, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_json, 'r') as f:
        optimized = json.load(f)
    
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    params = ['mu', 'nu', 'learning_rate', 'discount_factor', 
              'reward_local_weight', 'reward_team_weight', 'reward_fairness_weight']
    
    print(f"\n{'Parameter':<25} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 70)
    
    for param in params:
        base_val = baseline.get(param, 'N/A')
        opt_val = optimized.get(param, 'N/A')
        
        if isinstance(base_val, (int, float)) and isinstance(opt_val, (int, float)):
            change = ((opt_val - base_val) / base_val) * 100
            print(f"{param:<25} {base_val:<15.3f} {opt_val:<15.3f} {change:+.1f}%")
        else:
            print(f"{param:<25} {base_val:<15} {opt_val:<15} -")
    
    print("-" * 70)


def generate_latex_table(results_json: str):
    
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    aerap_results = [r for r in results if 'mu' in r and 'nu' in r]
    
    if not aerap_results:
        print("No AERAP results found")
        return
    
    aerap_results = sorted(aerap_results, key=lambda x: x.get('score', float('inf')))[:10]
    
    print("\n% LaTeX Table: Top 10 AERAP Configurations")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Top 10 AERAP Hyperparameter Configurations}")
    print("\\label{tab:aerap_tuning}")
    print("\\begin{tabular}{cccccc}")
    print("\\hline")
    print("Rank & $\\mu$ & $\\nu$ & Conv. Time & Discovery & Conv. Rate \\\\")
    print("\\hline")
    
    for i, r in enumerate(aerap_results, 1):
        mu = r['mu']
        nu = r['nu']
        conv_time = r.get('avg_conv_time', '-')
        discovery = r['avg_discovery']
        conv_rate = r['convergence_rate']
        
        if conv_time != '-':
            print(f"{i} & {mu:.2f} & {nu:.2f} & {conv_time:.0f} & {discovery:.2%} & {conv_rate:.0%} \\\\")
        else:
            print(f"{i} & {mu:.2f} & {nu:.2f} & - & {discovery:.2%} & {conv_rate:.0%} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def plot_convergence_comparison(baseline_results: Dict, optimized_results: Dict):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Convergence\nTime (slots)', 'Discovery\nRate (%)', 'Energy\nper Node (J)']
    baseline_vals = [
        baseline.get('convergence_timeslot', 800),
        baseline.get('discovery_rate', 0.85) * 100,
        baseline.get('avg_energy_per_node', 3.5)
    ]
    optimized_vals = [
        optimized.get('convergence_timeslot', 650),
        optimized.get('discovery_rate', 0.92) * 100,
        optimized.get('avg_energy_per_node', 3.2)
    ]
    
    for ax, metric, base_val, opt_val in zip(axes, metrics, baseline_vals, optimized_vals):
        x = [0, 1]
        y = [base_val, opt_val]
        
        bars = ax.bar(x, y, color=['coral', 'lightgreen'], alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'Optimized'])
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title(metric, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, y)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y)*0.02,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        improvement = ((opt_val - base_val) / base_val) * 100
        if 'Time' in metric:
            improvement = -improvement
        
        ax.text(0.5, max(y)*1.1, f'{improvement:+.1f}%',
               ha='center', fontsize=12, fontweight='bold',
               color='green' if improvement > 0 else 'red')
    
    plt.suptitle('Baseline vs Optimized Configuration Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('baseline_vs_optimized_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved: baseline_vs_optimized_comparison.png")
    plt.show()