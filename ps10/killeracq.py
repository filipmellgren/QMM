# Industry equilibrium with killer acquisitions and not taking w**theta = Y = 1 as given

# Keep first and last entry from entrydecisions. First is profit for incumbent, last is profit dfor entrant
# Calucllate expected profits if erntrant enters for icnumbet. This will be comapred to the price
# Before doing this. IMplement busection seach over pricesd W and Y

def industry_equilibrium(x, c_curvature, c_shifter, z_draws, prod_grid, probs, gamma, theta, competition, learning_rate):
	''' Simulate an industr, given prodcutivity draws.
	
	This implementation does not assume 
	Three stage game based on Edmond, Midrigan, Xu (2022), with a killer acquisition phase added
		0th period: Incumbent decides whether to acquire potential entrant
		1st period: Firms choose whether to pay an entry cost and enter
		2nd period: Static nested CES model
	
	INPUT :
	x : W**(-theta) * Y a guess of what this is. Want supply = demand so find this using bisection. Note, these are aggregates so find them in aggregate equilbirium.

	OUTPUT : 
	'''
 
	# First stage: Entry decisions
	# TODO: incumbent needs to make its decision before we do this

	n_incumbents = optimize.bisect(lambda x: entry_decisions(x, c_curvature, c_shifter, z_draws, prod_grid, probs, W, gamma, theta, Y, competition, learning_rate)[0], 1, 5000-1, xtol = 0.9) # TODO: check terminal condition
	
	marginal_entrant = n_incumbents + 1

	# Second stage: Static nested CES from lecture 1
	z_draws = z_draws[:int(marginal_entrant)]
	_, markups, prices, Pj = get_profits(W, z_draws, gamma, theta, Y, competition, learning_rate)
	loss, _, shares, elas,_ = markup_diff(markups, z_draws, gamma, theta, competition) 

	# Summary variables
	largest_firm = np.argmax(shares)
	share = shares[largest_firm]
	hhi = np.sum(np.asarray(shares)**2)
	Ptildej = price_aggregator(markups/z_draws, gamma) # Slide 42 L1
	# Industry markup Slide 45 lecture 1. "Industry markup is the cost weighted markup"
		# Slide 42 L1, true price index Pj = W * Ptildej -> Pj/W = Ptildej
	markupj = (Ptildej**(1 - gamma)) / np.sum(markups ** (-gamma) * z_draws**(gamma - 1))
	

	# demandij = prices **(-gamma) * Pj**(gamma-theta)*Y# Slide 14, lecture 2. Small firms
	# costs = demandij * prices / markups # TODO quantities Slide 4x lecture 1 ()
	# Slide 45 lecture 1
	Zj = (np.sum((markups/markupj)**(-gamma) * z_draws**(gamma - 1)))**(1/(gamma - 1))

	return(share, hhi, markupj, Ptildej, Zj)



def acquire():
	# Compute expected profits with a and without entrant for incumbent largest firm. 
	profit_acq, _, _, _ = get_profits(W, z_draws_acq, gamma, theta, Y, competition, learning_rate)
	profit_entry = , _, _, _ = get_profits(W, z_draws, gamma, theta, Y, competition, learning_rate)
	
	profit_diff = profit_acq - profit_entry
	return(profit_diff > entrant_benefit)

