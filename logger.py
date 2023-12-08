class Logger:
    '''freeform data collection for the plots'''
    def __init__(self):
        self.log = []

    def read(self):
        return self.log
    
    def update(self, dict):
        self.log.append(dict)

    def save(self):
        file='data.py' 
        with open(file, 'w') as f:
            f.write(f"data = {self.log}")

if __name__ == "__main__":
    log = Logger()
    log.update({"msg" : "hello world"})
    log.save()