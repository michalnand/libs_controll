import numpy



class SignalUnitStep:
    def __init__(self, samples_count, channels_count, amplitudes = [1.0]):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.amplitudes         = numpy.array(amplitudes, dtype=numpy.float32).reshape((1, len(amplitudes)))


    def sample_batch(self, batch_size = 32, batch_first = False):
        result = self.amplitudes*numpy.ones((self.samples_count, batch_size, self.channels_count), dtype=numpy.float32)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

class SignalSquare:
    def __init__(self, samples_count, channels_count, amplitudes = [1.0]):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.amplitudes         = numpy.array(amplitudes).reshape((1, len(amplitudes)))


    def sample_batch(self, batch_size = 32, batch_first = False):


        result = numpy.zeros((self.samples_count, batch_size, self.channels_count), dtype=numpy.float32)
        phases = numpy.random.randint(0, self.samples_count)

        amplitudes = self.amplitudes #*(2.0*numpy.random.rand(batch_size, self.channels_count) - 1.0)

        for n in range(self.samples_count):
            result[n]   = amplitudes*(n > phases)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

   


class SignalGaussianNoise:
    def __init__(self, samples_count, channels_count, amplitudes):
        self.samples_count      = samples_count
        self.channels_count     = channels_count
        self.amplitudes         = numpy.array(amplitudes).reshape((1, len(amplitudes)))


    def sample_batch(self, batch_size = 32, batch_first = False): 
        result = self.amplitudes*numpy.random.randn(self.samples_count, batch_size, self.channels_count)
           
        if batch_first:
            result = numpy.swapaxes(result, 0, 1)

        return result

