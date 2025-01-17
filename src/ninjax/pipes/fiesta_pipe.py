import os
import json
from typing import Callable
import numpy as np
from astropy.time import Time
import jax 
import jax.numpy as jnp

from jimgw.prior import Composite

import ninjax.pipes.pipe_utils as utils
from ninjax.pipes.pipe_utils import logger

from fiesta.inference.lightcurve_model import BullaLightcurveModel
from fiesta.utils import load_event_data

class FiestaPipe:
    
    def __init__(self,
                 config: dict,
                 outdir: str,
                 prior: Composite,
                 seed: int,
                 transforms: list[Callable]):
        
        self.config = config
        self.outdir = outdir
        self.complete_prior = prior
        self.seed = seed
        self.transforms = transforms
        
        # Preprocess the data a bit
        self.check_filters()
        self.load_data()
        
        # Set detection limit:
        self.set_detection_limit()
        
        # TODO: if we want an injection, then set up the injection here
        # ...
        
        # TODO: what if we combine KN and GRB? How is this done precisely?
        self.set_KN_lightcurve_model()
        self.set_GRB_lightcurve_model()
        
        self.set_EM_lightcurve_model()
        
        # Final setters to communicate with ninjax analysis:
        self.is_em_injection = eval(self.config["em_injection"])
        
    @property
    def filters(self):
        return self.config["filters"].split(",")
    
    @property
    def tmin(self):
        return float(self.config["tmin"])
    
    @property
    def tmax(self):
        return float(self.config["tmax"])
    
    @property
    def em_data_path(self):
        return str(self.config["em_data_path"])
    
    @property
    def trigger_time(self):
        return float(self.config["em_trigger_time"])
    
    @property
    def KN_model_name(self):
        return str(self.config["KN_model_name"])
    
    @property
    def KN_class_name(self):
        return str(self.config["KN_class_name"])
    
    @property
    def KN_model_dir(self):
        return str(self.config["KN_model_dir"])
    
    @property
    def GRB_model_name(self):
        return str(self.config["GRB_model_name"])
    
    @property
    def GRB_class_name(self):
        return str(self.config["GRB_class_name"])
    
    @property
    def GRB_model_dir(self):
        return str(self.config["GRB_model_dir"])
    
    def check_filters(self):
        logger.info(f"Checking the provided filters: {self.filters}")
        # TODO: implement this, check if it is an "allowed" filter or something?
        pass
    
    def load_data(self):
        logger.info(f"Loading EM data from {self.em_data_path}")
        if not os.path.exists(self.em_data_path):
            raise FileNotFoundError(f"EM data path {self.em_data_path} not found.")
        self.data = load_event_data(self.em_data_path)
        
        logger.info(f"EM data loaded, shown here:")
        
    def set_detection_limit(self):
        
        detection_limit = self.config["detection_limit"]
        if detection_limit.lower() == "none":
            detection_limit = 99999999.0
        
        self.detection_limit = float(detection_limit)
        
        logger.info(f"EM detection limit is set to {self.detection_limit}")
        return
        
    def set_KN_lightcurve_model(self):
        if self.KN_model_name.lower() == "none" or self.KN_model_name.lower() == "":
            logger.info("No KN model specified.")
            self.KN_lightcurve_model = None
            return
        
        if self.KN_class_name == "BullaLightcurveModel":
            logger.info("Setting a BullaLightcurveModel KN model.")
            self.KN_lightcurve_model = BullaLightcurveModel(self.KN_model_name,
                                                            self.KN_model_dir,
                                                            self.filters)
            
            return
    
    def set_GRB_lightcurve_model(self):
        # TODO: implement this
        pass
        # if self.GRB_model_name.lower() == "none" or self.GRB_model_name.lower() == "":
        #     logger.info("No GRB model specified.")
        #     self.GRB_lightcurve_model = None
        #     return
        

    def set_EM_lightcurve_model(self):
        
        # TODO: uhmmmm how? Doing it the hacky way for now. That is, this is only working for the KN
        self.EM_model = self.KN_lightcurve_model