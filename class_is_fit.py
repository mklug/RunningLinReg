'''Determines if the model has been fit which is 
required before calling other methods.  

We follow the sklearn method of doing this where
the attributes learned through fitting have names
that end with an underscore.  Thus, checking if 
fitting has occured amounts to checking if there
is an attribute ending in an underscore.  
'''


def fitted_attribute(att):
    return len(att) > 1 and att[-1] == '_'


def class_is_fit(model):
    return any(fitted_attribute(att) for att in vars(model))
