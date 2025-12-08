import configparser
import os

#this function was modified to be better for probing purposes
def read_arguments_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
 
    # 1. Initialize parser to ignore inline comments (strips #...)
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(file_path)

    # 2. Get the settings dictionary
    if 'settings' in config:
        out_dict = dict(config['settings'])
    else:
        # Fallback for missing header
        with open(file_path, 'r') as f:
            file_content = '[settings]\n' + f.read()
        config_string = configparser.ConfigParser(inline_comment_prefixes='#')
        config_string.read_string(file_content)
        out_dict = dict(config_string['settings'])

    # 3. SMART CONVERSION: Automatically convert types
    for key, value in out_dict.items():
        # Clean whitespace
        value = value.strip()
        
        # Check for Boolean
        if value.lower() == 'true':
            out_dict[key] = True
        elif value.lower() == 'false':
            out_dict[key] = False
        else:
            # Check for Number (Float or Int)
            try:
                if '.' in value or 'e' in value.lower():
                    out_dict[key] = float(value)
                else:
                    out_dict[key] = int(value)
            except ValueError:
                # Keep as string if it's not a number (e.g., "adam", "linear")
                out_dict[key] = value

    return out_dict

def _convert_arguments(settings):
    converted_settings = {}
    converted_settings['viz'] = settings.getboolean('viz')
    converted_settings['viz_every_iteration'] = False
    converted_settings['verbose'] = True
    converted_settings['method'] = settings['method']
    converted_settings['neurons_per_layer'] = settings.getint('neurons_per_layer')
    converted_settings['optimizer'] = settings['optimizer']

    converted_settings['batch_type'] = settings['batch_type']
    converted_settings['batch_size'] = settings.getint('batch_size')
    converted_settings['batch_time'] = 99999
    converted_settings['batch_time_frac'] = 99999

    converted_settings['init_lr'] = settings.getfloat('init_lr')
    converted_settings['weight_decay'] = settings.getfloat('weight_decay')
    converted_settings['dec_lr'] = False
    converted_settings['dec_lr_factor'] = 999

    converted_settings['cpu'] = settings.getboolean('cpu')
    converted_settings['val_split'] = settings.getfloat('val_split')
    converted_settings['noise'] = settings.getfloat('noise')
    converted_settings['epochs'] = settings.getint('epochs')

    converted_settings['solve_eq_gridsize'] = 100
    converted_settings['solve_A'] = False

    converted_settings['debug'] = False  
    converted_settings['output_dir'] = "output"
    converted_settings['normalize_data'] = settings.getboolean('normalize_data')  
    converted_settings['explicit_time'] = False
    converted_settings['relative_error'] = False

    converted_settings['pretrained_model'] = settings.getboolean('pretrained_model')   
    converted_settings['lr_range_test'] = False 
    converted_settings['scale_expression'] = settings.getfloat('scale_expression')
    converted_settings['log_scale'] = 'linear'
    converted_settings['init_bias_y'] = 0

    
    
    
    
    return converted_settings
