#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:50:49 2023

@author: nbadolo and Weizmann K.)
"""

import subprocess
import sys
from pathlib import Path
import os, shutil
import glob
from os.path import exists
from tqdm.auto import tqdm
import numpy as np



txt_folder = 'psf_id'
id_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/pyssed_log/DATA_CENTER/' +txt_folder+ '/txt_files/'
file_name = 'DC_stars_process_id.txt'
process_id = open("{}/{}".format(id_path, file_name), "w")
process_id.write("{}, {},{}\n".format('Star_name', 'Filter', 'Process_ID'))



path = Path('A_psf_files')


path, dirs, files = next(os.walk(path))



types = ['alone', 'both']


with tqdm(total=len(dirs)) as progress:
    
    for folder in dirs:
        for type in types:
            #sub_path, sub_dirs, sub_files = next(os.walk(f'{path}/{folder}/star/{type}'))
            sub_path, sub_dirs, sub_files = next(os.walk(f'{path}/{folder}/psf/{type}'))
            sub_folder_ar = np.zeros(len(sub_dirs))
               
            for sub_folder in sub_dirs:
                sh_file = f'{sub_path}/{sub_folder}/sphere_dl_script.sh'

                try:
                    if exists(f'{sh_file}'):
                        print('yes')
                        print(sh_file)
                        sub_folder_ar = str(sub_folder)
                        for fit in glob.glob(os.path.join(f'{sub_path}/{sub_folder}/', "*.fits")) : # make the list of '*.fits' fliles that already exist
                            print(fit)
                            
                            os.remove(f'{sub_path}/{sub_folder}/{fit}')                            # remove all the '*.fits' fliles before download news
                        ## change directory by moving to the '.sh' file loctaion
                        ## to make it excecutable  then run it
                        os.chdir(f'{sub_path}/{sub_folder}')
                        
                        # change  '.sh' mode 
                        process=subprocess.Popen(['chmod', '+x', 'sphere_dl_script.sh'])
                        process.wait()
                        
                        # run '.sh' file
                        command = subprocess.Popen(['./sphere_dl_script.sh'])
                        command.wait()
                        
                        ## move all the files in 'SPHERE_DC_DATA' to the same dir than '.sh' file 
                        
                        new_sub_path, new_sub_dirs, new_sub_files = next(os.walk('SPHERE_DC_DATA'))
                        
                        
                        #print(new_sub_dirs)
                        #print(new_sub_files)
                        #print(new_sub_path)
                        
                        # get the star process id
                        #ident_ar = np.zeros(len(new_sub_dirs))
                        process_ident = 0
                        for fd in new_sub_dirs:
                            print(fd)
                            print(fd[0])
                            print(len(fd))
                            for l in range(len(fd))  :
                                if fd[l:l+4] == 'ter_' :
                                    ident = fd[l+4:]
                                    process_ident = ident
                        process_id.write("{}, {}, {}\n".format(folder, sub_folder, 
                                                               process_ident))
                        print(process_ident)
                        ##copy all files in new_sub_files ( .pdf files)    
                        #for f in new_sub_files:
                        #    shutil.move(f'{new_sub_path}/{f}', './')
                            
                        #del f
                        
                        ## put all the new files in the list 
                        for new_sub_dir in new_sub_dirs:
                            new_files = os.listdir(f'{new_sub_path}/{new_sub_dir}')
                            
                            # move files one by one 
                            for f in new_files:
                                shutil.move(f'{new_sub_path}/{new_sub_dir}/{f}', './')

                        del f, new_sub_path, new_sub_dirs, new_sub_files
                        
                        
                        # delete the 'SPHERE_DC_DATA' dir 
                        shutil.rmtree('SPHERE_DC_DATA')
                        
                        #come back to the initial directory.
                        # current dir 
                        cwd = os.getcwd()

                        os.chdir(cwd.split(path)[0]+path+'/..')
                        
                        
                    else:
                        print(f'{sh_file} missing')
                        
                        
                except FileNotFoundError as e:
                    print(f'{sh_file} missing')
        
            
            del  sub_path, sub_dirs, sub_files
                    
    progress.update() 
                    
        