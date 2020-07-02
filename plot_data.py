def plot_the_data(self):

    #df = self.math_data[0]
    #X = self.do_the_maths()[1][0]
    #y = self.do_the_maths()[1][1]
    #y_pred = self.do_the_maths()[1][2]

    #df.groupby(['BidPrice']).size().round().plot(kind='line', color='white')
    #plt.ylabel('ExpectedRevenue')

    #plot_df = df.copy()

    # a=3.0, b=35.0, c=50.0, d=75.0


    # plt.hist(a.ExpectedRevenue, color='blue')
    # plt.hist(b.ExpectedRevenue, color='orange')
    # plt.hist(c.ExpectedRevenue, color='red')
    # plt.hist(d.ExpectedRevenue, color='green')
    """
         def _get_self_regression(__df: pd.DataFrame):
             _df = __df.sample(n=SAMPLE_SIZE)
             X = _df['ExpectedConversion']
             y = _df['ExpectedRevenue']
             _df['X'] = X
             _df['y'] = y
             print(_df)
             # Calculate the mean of X and y
             xmean = np.mean(_df['X'])
             ymean = np.mean(_df['y'])

             # Calculate the terms needed for the numerator and denominator of beta
             _df['xycov'] = (_df['X'] - xmean) * (_df['y'] - ymean)
             _df['xvar'] = (_df['X'] - xmean) ** 2

             # Calculate beta and alpha
             beta = _df['xycov'].sum() / _df['xvar'].sum()
             alpha = ymean - (beta * xmean)
             print(f'alpha = {alpha}')
             print(f'beta = {beta}')
             y_pred = alpha + (beta * X)

             return [X, y, y_pred]
         """


    """
    ax1 = plot_df.plot(kind='scatter', x=plot_df['BidPrice'] == 3.0,
                       y=plot_df.ExpectedRevenue, color='blue')
    ax2 = plot_df.plot(kind='scatter', x=35.0, y=b.ExpectedRevenue, color='orange')
    ax3 = plot_df.plot(kind='scatter', x=50.0, y=c.ExpectedRevenue, color='red')
    ax4 = plot_df.plot(kind='scatter', x=75.0, y=d.ExpectedRevenue, color='green')
    plt.legend(labels=['BidPrice', 'ExpectedRevenue'])
    plt.title('Relationship between BidPrice and ExpectedRevenue', size=24)
    plt.xlabel('BidPrice')
    plt.ylabel('ExpectedRevenue')
    """

    """
    plt.autoscale
    plt.figure(figsize=(5, 5))
    plt.plot(X, y_pred)  # regression line
    plt.plot(X, y, 'ro')  # scatter plot showing actual data
    plt.title('Actual vs Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    """

    #plt.autoscale()
    #plt.show()

    #plt.plot()
    """For further analytics uncomment print statements"""
    # print(a.describe())
    # print(b.describe())
    # print(c.describe())
    # print(d.describe())
    # print(a.sample(n=75).mean())
    # print(b.sample(n=75).mean())
    # print(c.sample(n=75).mean())
    # print(d.sample(n=75).mean())
    # print(f"At $3.0 \n{a.mean()}")
    # print(f"At $35.0 \n{b.mean()}")
    # print(f"At $50.0 \n{c.mean()}")
    # print(f"At $75.0 \n{d.mean()}")
