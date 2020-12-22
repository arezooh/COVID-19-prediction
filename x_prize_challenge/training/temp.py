iran_data = test_data[test_data['CountryCode'] == 'CAN'][6:-1]
iran_data.reset_index(inplace=True)
iran_data[target] = iran_data[target].astype(float)
fig = plt.figure()
plt.title('Canada')
plt.plot(iran_data['Date'].map(lambda x: datetime.strptime(str(x), '%Y%M%d')), iran_data['prediction'], '-r',
         label='prediction')
plt.plot(iran_data['Date'].map(lambda x: datetime.strptime(str(x), '%Y%M%d')), iran_data[target], '-b',
         label='real-value')
plt.legend()
plt.xticks(iran_data['Date'].map(lambda x: datetime.strptime(str(x), '%Y%M%d')))
fig.autofmt_xdate()
plt.show()
