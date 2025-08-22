import os
import argparse
import yaml
from alpaca_eval import evaluate_from_model

def load_config(default_config_path, overrides):
    with open(default_config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"model configs before overrides: {config}")

    for key, value in overrides.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    print(f"model configs after overrides: {config}")
    return config

def main():
    parser = argparse.ArgumentParser(description="Evaluate model using alpaca_eval")
    parser.add_argument('--model_configs', type=str, help='Path to the default config file')
    parser.add_argument('--override', action='append', help='Override config values, format: key=value')
    parser.add_argument('--annotators_config', type=str, help='Path to the annotators configuration or a dictionary.')
    parser.add_argument('--output_path', type=str, help='Path to save the output')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum number of instances to evaluate')
    args = parser.parse_args()
    print(f"args: {args}")
    
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            overrides[key] = yaml.safe_load(value)
    
    model_configs = load_config(args.model_configs, overrides)
    
    evaluate_from_model(model_configs=model_configs,
                        annotators_config=args.annotators_config,
                        output_path=args.output_path,
                        caching_path=os.path.join(args.output_path, "alpaca_eval_annotator_cache.json"),
                        precomputed_leaderboard=None,
                        is_cache_leaderboard=False,
                        is_return_instead_of_print=True,
                        max_instances=args.max_instances,)

if __name__ == "__main__":
    main()