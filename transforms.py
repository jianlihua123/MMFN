def linear(pic):
    #return pic*2.0-1.0
    return pic

class Stretch:
    def __call__(self, pic):
        return linear(pic)
