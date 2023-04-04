#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:50:49 2023

@author: nbadolo
"""

import subprocess
import sys
from pathlib import Path
import os 
from os.path import exists
from tqdm.auto import tqdm


# test = ['un','deux', 'trois']
# test_iter = iter(test)

# print(test)
# #print(test_iter)
# print(next(test_iter))
# print(next(test_iter))
# print(next(test_iter))

#%%
path = Path('/home/nbadolo/A_large_log/')
#%%
path, dirs, files = next(os.walk(path))

types = ['alone', 'both']
#%%

with tqdm(total=len(dirs)) as progress:
    
    for folder in dirs:
        for type in types:
            sub_path, sub_dirs, sub_files = next(os.walk(f'{path}/{folder}/star/{type}'))
    
            for sub_folder in sub_dirs:
                sh_file = f'{sub_path}/{sub_folder}/sphere_dl_script.sh'

                try:
                    if exists(sh_file):
                        print('yes')
                        print(f'{sh_file}')
                        process=subprocess.Popen(['chmod', '+x', sh_file])
                        process.wait()

                        command = subprocess.run('./', str(sh_file),
                            shell=True,
                            capture_output=True,
                        )
                        
                        #sys.stdout.buffer.write(command.stdout)
                        #sys.stderr.buffer.write(command.stderr)
                    
                    #else:
                        #print(f'{sh_file} missing')
                        
                        
                except FileNotFoundError as e:
                    print(f'{sh_file} missing')
    progress.update() 