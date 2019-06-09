import numpy as np

lam=3
REQUESTP1=[pow(lam,n)*pow(np.e,-lam)/np.math.factorial(n) for n in range(0,21)]
lam=4
REQUESTP2=[pow(lam,n)*pow(np.e,-lam)/np.math.factorial(n) for n in range(0,21)]
lam=3
RETURN1=[pow(lam,n)*pow(np.e,-lam)/np.math.factorial(n) for n in range(0,21)]
lam=2
RETURN2=[pow(lam,n)*pow(np.e,-lam)/np.math.factorial(n) for n in range(0,21)]

REQUEST=np.arange(0,21)

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11
ACTIONS=np.arange(-5,6)



def getReward(state,action,v):

	xx=-2*abs(action)	
	
	
	
	totalreward=0
	for i in range(POISSON_UPPER_BOUND):
		for j in range(POISSON_UPPER_BOUND):
			first,second=state
	
			first=min(first-action,MAXCAR)
			second=min(second+action,MAXCAR)

			actual_first_rental=min(first,i)
			actual_second_rental=min(second,j)
			reward=(actual_first_rental+actual_second_rental)*10

			first-=actual_first_rental
			second-=actual_second_rental
			
			first=min(first+3,MAXCAR)
			second=min(second+2,MAXCAR)
			newstate=(int(first),int(second))
			prob=REQUESTP1[i]*REQUESTP2[j]
			totalreward+=prob*(reward+gamma*v[newstate])
	
	return totalreward+xx

MAXCAR=20
v=np.zeros((MAXCAR+1,MAXCAR+1))
policy=np.zeros((MAXCAR+1,MAXCAR+1),dtype=int)
gamma=0.9

count=0


while True:

	while True:
		newv=np.copy(v)
		for i in range(MAXCAR+1):
			for j in range(MAXCAR+1):
				state=(i,j)
				newv[state]=getReward(state,policy[i][j],v)
		value_change = np.abs((newv - v)).sum()
		print('value change %f' % (value_change))
		v=newv
		if(value_change<1e-4):
			break
		
	newpolicy = np.zeros((MAXCAR+1,MAXCAR+1),dtype=int)
	for i in range(MAXCAR+1):
		for j in range(MAXCAR+1):

			xxx=[]
			for action in ACTIONS:
				if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
					xxx.append(getReward([i,j],action,v))
				else:
					xxx.append(-float('inf'))
			index=np.argmax(xxx)
			newpolicy[i][j]=ACTIONS[index]
	
	

	count+=1
	policy_change = (newpolicy != policy).sum()
	print('policy changed in %d states' % (policy_change))
	policy=newpolicy
	if policy_change==0:
		print(count)
		print(newv)
		print(newpolicy)
		break
	


	


