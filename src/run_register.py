import json
import os
import sys
from register import Register

def get_json_config():
    """Read config file and generate output folder (base directory)
    
    If no config file is given as argument, take the default config.json file
    
    Returns
    -------
    dict
    """
    try:
        config_file = sys.argv[1]
    except IndexError:
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    
    with open(config_file) as config_file:
        config = json.load(config_file)
    
    config['base_dir'] = config.get('base_dir', 'output')    
    os.makedirs(config['base_dir'], exist_ok=True)

    print('Configuration:')
    print(json.dumps(config, indent=4))
    
    return config

def run(config):
    """Initialize register, start chosen mode to run program and dump results 

    Parameters
    ----------
    config : dict
    """
    register = Register(config)

    if config['mode'] == 'train_classifier':
        results = register.train_classifier()
        if results and len(results)>1:
            register.dump_results(
                register.get_significance(
                    results,
                    metric='f_score',
                    best_det='total'),
                    file_name='significance_results')
             
    elif config['mode'] == 'train_linear_regressor':
        results = register.train_linear_regressor()
        if results and len(results)>1:
            register.dump_results(
                register.get_significance(results), file_name='significance_results')
               
    elif config['mode'] == 'classify':
        register.dump_results(register.classify(), file_name='classes')
        
    elif config['mode'] == 'score':
        register.dump_results(register.score(), file_name='scores')
        
    elif config['mode'] == 'cluster':
        register.dump_results(register.cluster(), file_name='clustering')

    elif config['mode'] == 'analyze':
        register.analyze()

    register.dump_config()    
    

if __name__ == '__main__':
    """Main function
    
    run register
    """
    
    print('###################### running register... ######################')

    run(get_json_config())
