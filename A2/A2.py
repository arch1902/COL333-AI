import sys
import json
sys.setrecursionlimit(10**6)


def dumper(out):
    dic = {}
    for nurse in range(len(out)):
        for day in range(len(out[0])):
            dic["N"+str(nurse)+"_"+str(day)] = out[nurse][day]
    with open("solution.json" , 'w') as file:
        json.dump(dic,file)
        file.write("\n")
    file.close()
    return


def partA(N,D,m,a,e):
  if 6*N < 7*(m+a+e):
    return
  rest_n = N-(m+a+e)
  if m > rest_n + a:
    return
  cons = {"N":N, "D":D, "M":m, "A":a, "E":e,"R": N-(a+m+e)}
  cur = {"M":0, "A":0, "E":0,"R":0}
  rests = [0 for i in range(N)]
  out  = [['' for i in range(D)] for j in range(N)]
  nurses = [i for i in range(N)]
  if backtrack_A((0,0),out,rests,cur,cons,nurses,1):
    # for i in out:
    #     print(i)
    dumper(out)
    return out
  else:
    return

def get_val(a):
  c = 0 
  for i in a:
    if i in ["M","E"]:
      c += 1
  return c

def cost(out,S):
  cost = 0
  for i in range(S):
    for j in out[i]:
      if j in ["E","M"]:
        cost += 1
  return cost

def partB(N,D,m,a,e,S):
  if 6*N < 7*(m+a+e):
    print("No Solution")
    return
  rest_n = N-(m+a+e)
  if m > rest_n + a:
    return
  cons = {"N":N, "D":D, "M":m, "A":a, "E":e,"R": N-(a+m+e)}
  cur = {"M":0, "A":0, "E":0,"R":0}
  rests = [0 for i in range(N)]
  out  = [['' for i in range(D)] for j in range(N)]
  nurses = [i for i in range(N)]
  global mx
  mx = -1
  if backtrack_A((0,0),out,rests,cur,cons,nurses,1):
    out = sorted(out, key=get_val,reverse = True)
    mx = max(mx,cost(out,S))
    dumper(out)
    print('Log Cost =',cost(out,S))
    # for i in out:
    #   print(i)
  cur = {"M":0, "A":0, "E":0,"R":0}
  rests = [0 for i in range(N)]
  out  = [['' for i in range(D)] for j in range(N)]
  if backtrack_B((0,0),out,rests,cur,cons,S):
    # for i in out:
    #   print(i)
    print('Log Cost =',cost(out,S))
    return out
  else:
    if mx == -1:
      print("No Solution")
    return

def backtrack_B(cor,out,rests,cur,cons,S):
  ''' cor = (x,y) xth day, yth Nurse
      x = [0,D] and y = [0,N]
      out = N x D;  N rows D columns
      out[i][j] => ith Nurse jth Day'''

  global mx

  if cor[1] < S:
    avail = ["M","E","R","A"]
  else:
    avail = ["R","A","M","E"]

  if cor[0] == cons["D"]:
    ans = out.copy()
    ans = sorted(ans, key=get_val,reverse = True)
    c = cost(ans,S)
    if c>mx:
      mx = c
      dumper(ans)
      print("Maximum updated to",c)
    return False

  if cur["M"] >= cons["M"]:
    avail.remove("M")
  if cur["E"] >= cons["E"]:
    avail.remove("E")
  if cur["A"] >= cons["A"]:
    avail.remove("A")
  if cur["R"] >= cons["R"]:
    avail.remove("R")
  if cor[0] >0 and out[cor[1]][cor[0]-1] in ["M","E"] and "M" in avail:
    avail.remove("M")

  if len(avail) == 0 :
    return False

  if cor[1] == cons["N"]-1:
    next = (cor[0]+1,0)
  else:
    next = (cor[0],cor[1]+1)

  j = -1
  mn = 10000
  for i in range(cons['N']):
    if rests[i]<mn:
      mn = rests[i]
      j = i
    
  for i in avail:
    cur[i]+=1
    out[cor[1]][cor[0]] = i
    is_rest = False
    if cor[0]%7 == 6:
      for k in range(7):
        if out[cor[1]][cor[0]-k] == "R":
          is_rest = True
          break
      if not is_rest:
        cur[i]-=1
        out[cor[1]][cor[0]] = ""
        continue
    if i=="R":
      rests[cor[1]] += 1
    if cor[1] == cons["N"]-1:
      cur = {"M":0, "A":0, "E":0,"R":0}
    if backtrack_B(next,out,rests,cur,cons,S):
      return True
    if i=="R":
      rests[cor[1]] -= 1
    if cor[1] == cons["N"]-1:
      cur = {"M":cons["M"], "A":cons["A"], "E":cons["E"],"R":cons["R"]}    
    out[cor[1]][cor[0]] = ''
    cur[i]-=1
  return False




def backtrack_A(cor,out,rests,cur,cons,nurses,j):
  ## cor = (x,y) xth day, yth Nurse
  ## x = [0,D] and y = [0,N]
  ## out = N x D;  N rows D columns
  ## out[i][j] => ith Nurse jth Day
  # print("XXXXXXXXX")
  # for i in out:
  #   print(i)
  
  avail = ["M","E","A","R"]

  if cor[0] == cons["D"]:
    return True
  if cur["M"] >= cons["M"]:
    avail.remove("M")
  if cur["E"] >= cons["E"]:
    avail.remove("E")
  if cur["A"] >= cons["A"]:
    avail.remove("A")
  if cur["R"] >= cons["R"]:
    avail.remove("R")
  if cor[0] >0 and out[cor[1]][cor[0]-1] in ["M","E"] and "M" in avail:
    avail.remove("M")

  # if j == cons["N"]-1:
  #   j = 0
  #   nurses = []
  #   for i in range(cons["N"]):
  #     if out[i][cor[0]] in ["M","E"]:
  #       nurses = [i] + nurses
  #     else:
  #       nurses.append(i)
  #   next = (cor[0]+1,nurses[j])
  # else:
  #   next = (cor[0],nurses[j+1])
  #   j += 1


### Prefer giving "R" to nurse not rested yet this week
  
  flag = False
  for day in range(cor[0]-cor[0]%7,cor[0]+1):
    if out[cor[1]][day]=="R":
      flag = True
      break
  if not flag:
    if "R" in avail:
      avail.remove("R")
      avail = ["R"] + avail

  for i in avail:
    prev_cur = cur.copy()
    cur[i]+=1
    out[cor[1]][cor[0]] = i

    is_rest = False
    if cor[0]%7 == 6:
      for k in range(7):
        if out[cor[1]][cor[0]-k] == "R":
          is_rest = True
          break
      if not is_rest:
        cur[i]-=1
        out[cor[1]][cor[0]] = ""
        continue

    prev_nurses = nurses.copy()
    if j == cons["N"]:
      # print("Xxxxxxxxxxxxxxxxxx")
      # for line in out:
      #   print(line)
      j = 0
      nurses = [i for i in range(cons["N"])]
      mn = 100000
      for k in range(cons["N"]):
        if out[k][cor[0]] in ["M","E"]:
          nurses.remove(k)
          nurses = [k] + nurses

      temp = nurses.copy()
      temp.reverse()

      for k in range(cons["N"]):
        mn = min(mn,rests[k])
      for k in temp:
        if rests[k]==mn:
          flag = False
          for day in range(cor[0]-cor[0]%7,cor[0]+1):
            if out[k][day]=="R":
              flag = True
              break
          if flag:
            continue
          nurses.remove(k)
          nurses = [k] + nurses

      next = (cor[0]+1,nurses[j])
      #print(nurses)
    else:
      next = (cor[0],nurses[j])
    j += 1
    
  
    if i=="R":
      rests[cor[1]] += 1
    if j == 1:
      cur = {"M":0, "A":0, "E":0,"R":0}
    

    if backtrack_A(next,out,rests,cur,cons,nurses,j):
      return True

    nurses = prev_nurses
    if i=="R":
      rests[cor[1]] -= 1
    if j == 1:
      j = cons["N"]+1
    j-=1
    out[cor[1]][cor[0]] = ''
    cur = prev_cur
  return False


if __name__ == "__main__":
    filename = sys.argv[1]
    file_ = open(filename,"r")
    lines = file_.readlines()
    file_.close()
    numParameters = len(lines[0].split(","))
    dumper([[]])
    if numParameters == 5:
        for ii in lines[1:]:
            data = ii.split(",")
            N = int(data[0])
            D = int(data[1])
            m = int(data[2])
            a = int(data[3])
            e = int(data[4])
            partA(N,D,m,a,e)
    elif numParameters == 7:
        for ii in lines[1:]:
            data = ii.split(",")
            N = int(data[0])
            D = int(data[1])
            m = int(data[2])
            a = int(data[3])
            e = int(data[4])
            S = int(data[5])
            T = int(data[6])
            partB(N,D,m,a,e,S)


    





    