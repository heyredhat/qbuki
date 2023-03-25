# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['foo', 'say_hello', 'HelloSayer']

# %% ../nbs/00_core.ipynb 4
def foo(): 
    "What the"
    return "What"

# %% ../nbs/00_core.ipynb 9
def say_hello(what):
    print(what)
    
class HelloSayer:
    "Say hello to `to` using `say_hello`"
    def __init__(self, to): self.to = to
        
    def say(self):
        "Do the saying"
        return say_hello(self.to)
show_doc(HelloSayer.say)
