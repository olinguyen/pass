#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys, os, time, datetime, json
import pymongo
import random


#Variables that contains the user credentials to access Twitter API
access_token = "1969348759-cSBoFcNzP9uNqlkTWjXTHNwqSVxpFQgx7bi9E5f"
access_token_secret = "qBA15hWersfXZnl72d5BJEgtReL3AJC5RvEjPvNVSMdu5"
consumer_key = "fS304oeTDngBgEOvZWdipflzh"
consumer_secret = "vhJBVAwdQCAQPdSBxLZWRTZSjLunZKJCdGBPMGwR5TKBOKB6bn"



#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)

####################
# Dont print the tweet
# load as it comes in
# filter out for geo-tagged in Canada
# save batches of 5 mins each in json format
# use shell script to automate search for multiple keywords
####################

class LimitListener(StreamListener):
    def __init__(self, time_limit=5*60):
        self.start_time = time.time()
        self.limit = time_limit

        now = datetime.datetime.utcnow().strftime("%Y.%m.%d.%H.%M.%s")

        #self.saveFile = open('RawTweets/%s_tws%s.json' %('_'.join(sys.argv[1:]).replace(' ','.'),now), 'a')

        #no keyword
        self.saveFile = open('RawTweets/%s_tws%s.json' %('Canada',now), 'a')

        super(LimitListener, self).__init__()

    def on_data(self, data):

        if (time.time() - self.start_time) < self.limit:
            self.saveFile.write(data)
            #self.saveFile.write('\n') # no need for extra space
            return True
        else:
            self.saveFile.close()
            return False

    def on_error(self, status):
        print(status)




# Put in MongoDB
class MongoDBListener(StreamListener):

    def on_data(self, data):

        #print(data)

        try:
            temp = json.loads(data)
            temp['random_number'] = random.random()
            if 'text' in temp:
                #print('Good Tweet')
                if temp['place']['country_code'] == 'CA':
                    if temp['truncated']:
                      temp['text'] = temp['extended_tweet']['full_text']

                    col.insert_one(temp)
                    #print(data)
                    #print('CANADA')
        except:
            pass


        if col.count() % 1000 == 0:
          print("# tweets in the database:", col.count())
        return True


    def on_error(self, status):
        print(status)

####################

if __name__ == '__main__':



    l = MongoDBListener()

    col = pymongo.MongoClient()["tweets"]["april"]


    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener=l)

    # Bounding box http://boundingbox.klokantech.com/
    # track and locations are OR per https://stackoverflow.com/questions/25739073/how-to-get-location-wise-tweets-using-tweepy-for-streaming-api


    # no keyword
    #geo only

    i = 1
    while True: # brute force to ignore exceptions
        try:
            stream.filter(locations=[-140.625,49.0091,-46.9336,70.3187
                                      ,-95.2,45.14,-70.81,53.65
                                      ,-83.18,42.68,-73.05,49.0
                                      ,-83.2562,41.5995,-77.2537,44.7011
                                      ,-80.8173,42.7211,-78.836,43.3508
                                      ,-82.38,45.03,-71.53,48.56
                                      ,-73.37,44.98,-67.76,49.46
                                      ,-67.73,43.33,-40.9,49.26 ])#, async = True)

        except:
            time.sleep(10)
            continue
