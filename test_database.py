import sqlite3

connection=sqlite3.connect("login_master")
crsr=connection.cursor()
try:
    crsr.execute('create table login1 (user varchar(30) primary key,password varchar(30))')
except:
    pass


def add_to_database(user,password="",new_old=0,change=0,new_password=""):
    if change==1:
        connection.execute("""update login1
set password= ?
where user= ?""",(new_password,user,))
        connection.commit()
        return("a")
    else:
        if new_old==0:#for login
            a=connection.execute("select password from login1 where user=?",(user,)).fetchall()
            print(a[0][0])
            if password==a[0][0]:
                return("fg")
                #login success
            else:
                pass
                #retry
        elif new_old==1:#for new user
            connection.execute("insert into login1 values(?,?)",(user,password))
            connection.commit()
print(add_to_database("admin","admin2"))
