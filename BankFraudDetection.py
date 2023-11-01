import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

csv = pd.read_csv("BankFraud.csv",sep=",")

df = pd.DataFrame(csv)

print("Veri setinin ilk 5 satırı")
print(df.head())

print(df.info())

print("\n")



print("Kategoriye Göre Ödeme Miktarının ve Dolandırıcılık Olma Durumunun Ortalaması")
print(df.groupby('category')[['amount','fraud']].mean())
print("\n\n\n")


print("Yaşa Göre Dolandırıcılık Yüzdesi")
print((df.groupby('age')['fraud'].mean()*100).reset_index().rename(columns={'age':'Age','fraud' : 'Fraud Percent'}).sort_values(by='Fraud Percent'))



print("\n\n")


print("step değişkeninin eşsiz değer sayısı       : ",df.step.nunique())
print("customer değişkeninin eşsiz değer sayısı   : ",df.customer.nunique())
print("age değişkeninin eşsiz değer sayısı        : ",df.age.nunique())
print("gender değişkeninin eşsiz değer sayısı     : ",df.gender.nunique())
print("zipcodeOri değişkeninin eşsiz değer sayısı : ",df.zipcodeOri.nunique())
print("merchant değişkeninin eşsiz değer sayısı   : ",df.merchant.nunique())
print("zipMerchant değişkeninin eşsiz değer sayısı: ",df.zipMerchant.nunique())
print("category değişkeninin eşsiz değer sayısı   : ",df.category.nunique())
print("amount değişkeninin eşsiz değer sayısı     : ",df.amount.nunique())
print("fraud değişkeninin eşsiz değer sayısı      : ",df.fraud.nunique())



print("\n\n")


df_reduced = df.drop(["zipcodeOri","zipMerchant"],axis=1)
print(df_reduced.head())


print("\n\n")


col_categorical = df_reduced.select_dtypes(include= ['object']).columns
print(type(col_categorical))
for col in col_categorical:
    df_reduced[col] = df_reduced[col].astype('category')

df_reduced[col_categorical] = df_reduced[col_categorical].apply(lambda x: x.cat.codes)
print(df_reduced.head(5))



print("\n\n")



bagimsiz_degiskenler = df_reduced.drop(["fraud"],axis=1)
hedef_degisken = df_reduced["fraud"]


x_train,x_test,y_train,y_test = train_test_split(bagimsiz_degiskenler,hedef_degisken,test_size=0.25,random_state=0)

kararAgaci = DecisionTreeClassifier()                                    #Karar Ağacı (Decision Tree) algoritması
kararAgaci.fit(x_train,y_train)                                          #Modelin eğitilmesi.
kararAgaci_tahmin = kararAgaci.predict(x_test)                           #Eğitilen modelin dolandırıcılık durumu tahmini.
kararAgaci_skor = accuracy_score(y_test,kararAgaci_tahmin)               #Doğruluk skoru

lojistikRegresyon = LogisticRegression(max_iter=1000)                                 #Lojistik Regresyon(Logistic Regression) algoritması.
lojistikRegresyon.fit(x_train,y_train)                                   #Modelin eğitilmesi.
lojistikRegresyon_tahmin = lojistikRegresyon.predict(x_test)             #Eğitilen modelin dolandırıcılık durumu tahmini.
lojistikRegresyon_skor = accuracy_score(y_test,lojistikRegresyon_tahmin) #Doğruluk skoru.

gnb = GaussianNB()                                                       #Gaussian Naive Bayes algoritması.
gnb.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
gnb_tahmin = gnb.predict(x_test)                                         #Eğitilen modelin dolandırıcılık durumu tahmini.
gnb_skor = accuracy_score(y_test,gnb_tahmin)                             #Doğruluk skoru.

knn = KNeighborsClassifier( metric='minkowski')                          #K-En Yakın Komşu(KNN) algoritması.
knn.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
knn_tahmin = knn.predict(x_test)                                         #Eğitilen modelin dolandırıcılık durumu tahmini.
knn_skor = accuracy_score(y_test,knn_tahmin)                             #Doğruluk skoru.

rastgeleOrman = RandomForestClassifier()                                 #Rastgele Ormanlar(Random Forest) algoritması.
rastgeleOrman.fit(x_train,y_train)                                       #Modelin eğitilmesi.
rastgeleOrman_tahmin = rastgeleOrman.predict(x_test)                     #Eğitilen modelin dolandırıcılık durumu tahmini.
rastgeleOrman_skor = accuracy_score(y_test,rastgeleOrman_tahmin)         #Doğruluk skoru.

svm = SVC()                                                              #Destek Vektör Makineleri(SVM) algoritması.
svm.fit(x_train,y_train)                                                 #Modelin eğitilmesi.
svm_tahmin = svm.predict(x_test)                                         #Eğitilen modelin dolandırıcılık durumu tahmini.
svm_skor = accuracy_score(y_test,svm_tahmin)                             #Doğruluk skoru.

xgboost = xgb.XGBClassifier()                                            #XGBoost algoritması.
xgboost.fit(x_train, y_train)                                            #Modelin Eğitilmesi.
xgboost_tahmin = xgboost.predict(x_test)                                 #Eğitilen modelin dolandırıcılık durumu tahmini.
xgboost_skor = xgboost.score(x_test,xgboost_tahmin)                      #Doğruluk skoru.

#Doğruluk değerlerinin yazdırılması.
print("\n\nAlgoritmaların Doğruluk Değerleri")
print(pd.DataFrame({"Algoritmalar":["Decision Tree","Logistic Regression","GNB","KNN","Random Forest","SVM","XGBoost"],
              "Doğruluk":[kararAgaci_skor, lojistikRegresyon_skor, gnb_skor, knn_skor, rastgeleOrman_skor,svm_skor,xgboost_skor]}))

models = []
models.append(('Logistic Regression', LogisticRegression(max_iter=1000)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGBoost', XGBClassifier()))

results_boxplot = []
names = []
results_mean = []
results_std = []
p,t = bagimsiz_degiskenler.values, hedef_degisken.values.ravel()
for name, model in models:
    cv_results = cross_val_score(model, p,t, cv=10)                       
    results_boxplot.append(cv_results)
    results_mean.append(cv_results.mean())                                
    results_std.append(cv_results.std())                                  
    names.append(name)


print("\n\n Cross Validation sonrası algoritmaların doğruluk ortalaması ve doğruluk standart sapması.")
print(pd.DataFrame({"Algoritmalar":names,"Doğruluk Ortalaması":results_mean,
              "Doğruluk Standart Sapması":results_std}))



