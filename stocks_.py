from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import quandl , math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from googlefinance import getQuotes
import json


start = datetime.datetime(2010,1,5)
end = (datetime.datetime.now().date() - datetime.timedelta(days=1))


def get_stock_info(symbol):
    ''' Gets finanical data for a given stock '''
    df = data.DataReader(name=symbol,data_source="google",start=start,end=end)
    df['Pct_Change O/C'] = round(((df.Close-df.Open)/df.Open)*100,2)
    df['Pct_Change O/L'] = round(((df.Low-df.Open)/df.Open)*100,2)
    df['Volume'] = round(df['Volume']/1000000,2)

    return df

def get_avg_change_open_low(df):
    ''' gets the avg pcnt change btwn open and low '''
    answer = df['Pct_Change O/L'].mean()
    
    return answer

#This tells us what the average intra-day range of a given stock in %pcnt.
def get_avg_change_open_close(df):
    ''' gets the avg pcnt change btwn open and close '''
    answer = df['Pct_Change O/C'].mean()
    
    return answer

def show_histo(symbol):
    ''' will show a histogram of day to day percent change Open to close '''
    df = data.DataReader(name=symbol,data_source="yahoo",start=start,end=end) #built in datareader, pulls in stock infro from data_source, set start time adn end time
    df['Pct_Change O/C'] = round(((df.Close-df.Open)/df.Open)*100,2) # creates a new col, the pct change from two diff col
    df['Pct_Change O/L'] = round(((df.Low-df.Open)/df.Open)*100,2) #
    oc_pct = df['Pct_Change O/C'].as_matrix()
    print (oc_pct)
    axes =plt.axes()
    plt.hist(oc_pct,30)
    plt.title(symbol + " avg Percentage Change Open To Close" + str(get_avg_change_open_low(df)))
    axes.set_xticks(list(range(-4,5)))
    plt.show()

    return df

def corr(stock1,stock2):
    '''Finds the correaltion coefficient between two diffifrent stocks'''
    stock1 = get_stock_info(stock1)
    stock2 = get_stock_info(stock2)

    answer = np.corrcoef(stock1['Close'],stock2['Close'])
    return answer

def make_csv(df,path,name='star'):
    name = "%s.csv" %(name)
    path = path + name
    #path = ("/Users/star/Desktop/Data_Udemy/%s.csv") %(stock)
    df.to_csv(path) 
    #print ("donezo")
    return df

def get_SPY():
	df_SPY = get_stock_info("SPY")
	df_SPY.rename(columns={'Pct_Change O/C':'S&P Performance'},inplace=True)
	df_SPY.rename(columns={'Volume':'S&P Volume'},inplace=True)
	#print (df_SPY.head(5))
	return df_SPY

def create_dataframes(df_1,df_SPY):
	df_combined = pd.concat([df_1,df_SPY['S&P Performance']],axis=1)
	df_combined = pd.concat([df_combined,df_SPY['S&P Volume']],axis=1)
	df_combined.dropna(how='any',inplace=True)
	df_combined['Prediction']=np.where(df_combined.Open<df_combined.Close,2,1)
	df_combined['Prediction'] = df_combined['Prediction']
	df_combined['Label'] = df_combined['Prediction']
	del df_combined['Prediction']
	final = df_combined
	return final


def prediction(df):
	#df = starsky
	features = df[['Open', 'High', 'Low', 'Volume', 'Pct_Change O/C', 'Pct_Change O/L', 'S&P Performance', 'S&P Volume']] #put close back
	response = df['Label']
	response = response.as_matrix()
	features = features.as_matrix()
	X = features
	y = response
	knn = KNeighborsClassifier(n_neighbors=5)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
	knn.fit(X_train,y_train)
	y_pred= knn.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	accuracy = round(accuracy*100)
	return (accuracy)

def prediction_2(symbol):
	''' gets a stock symbol and returns accuracy '''
	#df = starsky
	stock_df = get_stock_info(symbol)
	df = create_dataframes(stock_df,get_SPY())
	features = df[['Open', 'High', 'Low', 'Volume', 'Pct_Change O/C', 'Pct_Change O/L', 'S&P Performance', 'S&P Volume']] #put close back
	response = df['Label']
	response = response.as_matrix()
	features = features.as_matrix()
	X = features
	y = response
	knn = KNeighborsClassifier(n_neighbors=5)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
	knn.fit(X_train,y_train)
	y_pred= knn.predict(X_test)
	accuracy = metrics.accuracy_score(y_test,y_pred)
	accuracy = round(accuracy*100)
	return (accuracy)


def run_algo():
	''' checks algorithm accuracy across all stocks in S&P'''

	all_tkrs = pd.read_csv("/Users/star/Desktop/stock_symbols/S&P.csv")
	tkr_list = all_tkrs.symbol.tolist()
	pred_list = {}
	for tickers in tkr_list[0:100]:
		try:
			df = create_dataframes(get_stock_info(tickers),get_SPY())
			answer = prediction(df)
			answer = round(answer*100)
			pred_list.update({tickers: answer})
			print (tickers + " Is Done")

		except:
			print (tickers + " Dosen't Exist")
			pass

	final = pd.DataFrame.from_dict(pred_list, orient='index').rename(columns={0:'accuracy'})
	final.sort_values('accuracy', ascending=False, inplace=True)
	print (final.head(5))
	return final

#--------------------------------Turn the above into a class Object ritented negro-------------------------------------------------------#

class Stock(object):
	spy = get_SPY()

	def __init__(self,symbol):
		self.symbol = symbol
		self.data = get_stock_info(self.symbol)

	def get_avg_change_open_low(self):
		''' Gets finanical data for a given stock '''
		answer = round(self.data['Pct_Change O/L'].tail(100).mean(),2)
		return answer

	def get_avg_change_open_close(self):
		''' gets the avg pcnt change btwn open and low '''
		answer = round(self.data['Pct_Change O/C'].tail(100).mean(),2)
		return answer

	def show_histo(self):
		''' will show a histogram of day to day percent change Open to close '''
		df = data.DataReader(name=self.symbol,data_source="google",start=start,end=end) #built in datareader, pulls in stock infro from data_source, set start time adn end time
		df['Pct_Change O/C'] = round(((df.Close-df.Open)/df.Open)*100,2) # creates a new col, the pct change from two diff col
		df['Pct_Change O/L'] = round(((df.Low-df.Open)/df.Open)*100,2) #
		oc_pct = df['Pct_Change O/C'].as_matrix()
		print (oc_pct)
		axes =plt.axes()
		plt.hist(oc_pct,30)
		plt.title(str(self.symbol).upper() + " avg Percentage Change Open To Close" + str(get_avg_change_open_low(df)))
		axes.set_xticks(list(range(-4,5)))
		plt.show()

		return df

	def corr(self,stock2):
		'''Finds the correaltion coefficient between two diffifrent stocks'''
		stock1 = self.data
		stock2 = get_stock_info(stock2)

		answer = round(np.corrcoef(stock1['Close'],stock2['Close'])[0][1]*100,2)
		return answer

	def prediction(self):
		''' gets a stock symbol and returns accuracy '''
		stock_df = self.data
		df = create_dataframes(stock_df,self.spy)
		features = df[['Open', 'High', 'Low', 'Volume', 'Pct_Change O/C', 'Pct_Change O/L', 'S&P Performance', 'S&P Volume']] #put close back
		response = df['Label']
		response = response.as_matrix()
		features = features.as_matrix()
		X = features
		y = response
		knn = KNeighborsClassifier(n_neighbors=5)
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
		knn.fit(X_train,y_train)
		y_pred= knn.predict(X_test)
		accuracy = metrics.accuracy_score(y_test,y_pred)
		accuracy = round(accuracy*100)
		
		return (accuracy)

	def largest_drop(self):
		answer = self.data.sort_values('Pct_Change O/C', ascending=True).head(3)
		return answer

	def average(self):
		# print self.data
		pass

	def line_chart(self):
		pass

	def rt_data(self):

		rt_data = getQuotes(self.symbol)
		info = []
		info.append(rt_data[0]['StockSymbol'])
		info.append(rt_data[0]['LastTradePrice'])
		info.append(rt_data[0]['LastTradeTime'])

		return info

#----------------------#

# baidu = Stock("BIDU")
akrx = Stock("AKRX")
nlsn = Stock("NLSN")
tdoc = Stock("TDOC")
gspc =Stock("GSPC")

print (gspc.rt_data())
# print (goog.data.tail(5))
# print (goog.get_avg_change_open_close())
#------------------------------------------------------predcitions-----------------------------------------------------#

all_tkrs = pd.read_csv("/Users/star/Desktop/stock_symbols/S&P.csv")
tkr_list = all_tkrs.symbol.tolist()
'''
pred_list = {}
for tickers in tkr_list[0:10]:
	try:
		df = create_dataframes(get_stock_info(tickers),get_SPY())
		#df.name = tickers
		answer = prediction(df)
		answer = round(answer,2)
		#make_csv(df,"/Users/star/Desktop/Python_CRISIL/AI_Stock_CSVs/",tickers)
		#final = t
		#final = dict([(tickers, answer)])
		pred_list.update({tickers: answer})
		#pred_list.append(final)
		print (tickers + " Is Done")

	except:
		print (tickers + " Dosen't Exist")
		pass


ash = pd.DataFrame.from_dict(pred_list, orient='index').rename(columns={0:'accuracy'})
ash.rename(columns={'0':'accuracy'}, inplace=True)
# print (ash.sort_values('accuracy', ascending=False).head(20))

# print (ash.index.tolist())
'''

#------------------------------------------------------predcitions-----------------------------------------------------#
