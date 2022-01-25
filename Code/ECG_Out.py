from ecgTools import ecgTools
ecgTools = ecgTools(250,20)

# ecgIn = ecgTools.getPhysionetData("18184",'nsrdb/',33000,38000)
ecgTools.readFromSensor()
ecgRaw = ecgTools.getRawData('ecgOut.txt')
ecgFFTa = ecgTools.fftFilt(ecgRaw,63)

ecgButter = ecgTools.butterFilt(ecgFFTa,1, 1, 'high')
ecgSma = ecgTools.sma(ecgRaw,3)
ecgEma = ecgTools.ema(ecgRaw,3,2)
# ecgFFTb = ecgTools.fftFilt(ecgButter,63)
pred = ecgTools.predict(ecgTools.reshape(ecgButter),'/home/pi/Desktop/ECG-Analysis/py-spidev-master/models/11_4_longrun/cp.ckpt')
print(pred)

ecgTools.ecgPlot([ecgRaw,ecgFFTa],0,2500,["Raw", "RFFT"],('Classification:',pred))


