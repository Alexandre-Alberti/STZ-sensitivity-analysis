#import math
import numpy as np
#import pandas as pd
import scipy
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import streamlit as st
#import sys
#from streamlit import cli as stcli
#from PIL import Image
from numpy import random 
import matplotlib.pyplot as plt 
# Funções e definições anteriores

def main():
    col1, col2, col3 = st.columns(3)
    
    st.image("foto.png")
    #foto = Image.open('foto.png')
    #col2.image(foto, use_column_width=True)

    st.title('Política STZ - Análise de Sensibilidade')

    menu = ["Aplicação", "Informação", "Website"]
    choice = st.sidebar.selectbox("Selecione aqui", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insira os valores dos parâmetros abaixo")
        
        Beta = st.number_input('Beta - parâmetro de forma da distribuição de probabilídade de Weibull para o tempo até a falha',format="%.7f", step = 0.0000001)
        betaimprec = st.number_input('Imprecisão na estimativa de Beta (%)',format="%.7f", step = 0.0000001)
        betaimprec = betaimprec/100
        Eta = st.number_input('Eta - parâmetro de escala da distribuição de probabilídade de Weibull para o tempo até a falha',format="%.7f", step = 0.0000001)
        etaimprec = st.number_input('Imprecisão na estimativa de Eta (%)',format="%.7f", step = 0.0000001)
        etaimprec = etaimprec/100
        Lbda = st.number_input('Lambda - taxa de chegada de oportunidades para manutenção',format="%.7f", step = 0.0000001)
        lbdaimprec = st.number_input('Imprecisão na estimativa de Lambda (%)',format="%.7f", step = 0.0000001) 
        lbdaimprec = lbdaimprec/100
        Cp = st.number_input('Cp - custo de substituição preventiva em T (programada)',format="%.7f", step = 0.0000001)
        cpimprec = st.number_input('Imprecisão na estimativa de Cp (%)',format="%.7f", step = 0.0000001) 
        cpimprec = cpimprec/100
        Cv = st.number_input('Cv - custo de substituição preventiva em Z (prorrogada)',format="%.7f", step = 0.0000001)
        cvimprec = st.number_input('Imprecisão na estimativa de Cv (%)',format="%.7f", step = 0.0000001)
        cvimprec = cvimprec/100
        Co = st.number_input('Co - custo de substituição preventiva antecipada por oportunidade',format="%.7f", step = 0.0000001)
        coimprec = st.number_input('Imprecisão na estimativa de Co (%)',format="%.7f", step = 0.0000001) 
        coimprec = coimprec/100
        Cw = st.number_input('Cw - custo de substituição preventiva em oportunidade posterior a T',format="%.7f", step = 0.0000001)
        cwimprec = st.number_input('Imprecisão na estimativa de Cw (%)',format="%.7f", step = 0.0000001)
        cwimprec = cwimprec/100
        Cf = st.number_input('Cf - custo de substituição corretiva',format="%.7f", step = 0.0000001) 
        cfimprec = st.number_input('Imprecisão na estimativa de Cf (%)',format="%.7f", step = 0.0000001)
        cfimprec = cfimprec/100
        P = st.number_input('P - probabilidade de impedimento para substituição preventiva na data programada',format="%.7f", step = 0.0000001)
        pimpre = st.number_input('Imprecisão na estimativa de P (%)',format="%.7f", step = 0.0000001)
        pimpre = pimpre/100 

        st.subheader("Insira os valores das variáveis de decisão da política de manutenção (política STZ)")
        S = st.number_input('S - data de abertura da janela para aproveitamento de oportunidades')
        T= st.number_input('T - data programada para a substituição preventiva')    
        Z= st.number_input('Z - data limite para a substituição preventiva em caso de prorrogação')
        y = (S, T, Z)
        
        st.subheader("Clique no botão abaixo para rodar esse aplicativo:")
        
        botao = st.button("Obtenha os valores")
        if botao: 
            resultados = [] 
            # Definições das funções
            def fx(x): 
                f = (beta/eta)*((x/eta)**(beta-1))*np.exp(-(x/eta)**beta) 
                return f 
            def Fx(x):
                return 1 - np.exp(-(x/eta)**beta) 
            def Rx(x): 
                return 1 - Fx(x)
                    
            def fh(h):
                return lbda*np.exp(-(lbda*h))
            def Fh(h):
                return 1 - np.exp(-(lbda*h)) 
            def Rh(h): 
                return 1- Fh(h) 

            def objetivo(y):
                S, T, Z = y  # Corrigindo a desestruturação das variáveis
                #CASO 1
                def P1(S):
                    return Fx(S)
                def C1(S):
                    return cf*P1(S)
                def V1(S):
                    return (quad(lambda x: x*fx(x), 0, S)[0])  
    
                #CASO 2
                def P2(S,T):
                    return Rh(T-S)*(Fx(T) - Fx(S)) + (dblquad(lambda x, h: fh(h)*fx(x), 0, T-S, lambda h: S, lambda h: S+h)[0])
                def C2(S,T):
                    return cf*P2(S,T)
                def V2(S,T):
                    return Rh(T-S)*(quad(lambda x: x*fx(x), S, T)[0])+ (dblquad(lambda x, h: x*fh(h)*fx(x), 0, T-S, lambda h: S, lambda h: S+h)[0])
                
                #CASO 3
                def P3(S,T,Z):
                    return p*Rh(Z-S)*(Fx(Z)-Fx(T)) + p*(dblquad(lambda x, h: fh(h)*fx(x), T-S, Z-S, lambda h: T, lambda h: h+S)[0])
                def C3(S,T,Z):
                    return cf*P3(S,T,Z)
                def V3(S,T,Z):
                    return  p*Rh(T-S)*(quad(lambda x: x*fx(x), T, Z)[0]) + p*(dblquad(lambda x, h: x*fh(h)*fx(x), T-S, Z-S, lambda h: T, lambda h: h+S)[0])
                
                #CASO 4
                def P4(S,T):
                    return (quad(lambda h: fh(h)*Rx(S+h), 0, T-S)[0])
                def C4(S,T):
                    return co*P4(S, T)
                def V4(S,T):
                    return (quad(lambda h: (S+h)*fh(h)*Rx(S+h), 0, T-S)[0])
    
                #CASO 5
                def P5(S,T,Z):
                    return p*(quad(lambda h: fh(h)*Rx(S+h), T-S, Z-S)[0])
                def C5(S,T,Z):
                    return cw*P5(S, T, Z)
                def V5(S,T,Z): 
                    return p*(quad(lambda h: (S+h)*fh(h)*Rx(S+h), T-S, Z-S)[0])
    
                #CASO 6
                def P6(S,T):
                    return (1-p)*Rh(T-S)*Rx(T) 
                def C6(S,T):
                    return cp*P6(S, T)
                def V6(S,T):
                    return T*P6(S, T)
    
                #CASO 7 
                def P7(S,T,Z):
                    return p*Rh(Z-S)*Rx(Z)
                def C7(S,T,Z):
                    return cv*P7(S, T, Z)
                def V7(S,T,Z):
                    return Z*P7(S, T, Z)

                SOMA_PROB=P1(S)+P2(S,T)+P3(S, T, Z)+P4(S, T) + P5(S, T, Z) + P6(S, T)+P7(S, T, Z)
                SOMA_CUST=C1(S)+C2(S,T)+C3(S, T, Z)+C4(S, T) + C5(S, T, Z) + C6(S, T)+C7(S, T, Z)
                SOMA_VIDA=V1(S)+V2(S,T)+V3(S, T, Z)+V4(S, T) + V5(S, T, Z) + V6(S, T)+V7(S, T, Z)

                TAXA_CUSTO=SOMA_CUST/SOMA_VIDA
                return TAXA_CUSTO
            
            x0 = [0.9, 1.0, 2.0]

            def cond1(y):
                return y[1]-y[0] # T >= S

            def cond2(y):
                return y[2]-y[1] # Z >= T

            Lista_test = []
            for i in range(0, 20): #LEMBRAR DE COLOCAR OS 400
                beta = random.uniform(Beta * (1 - betaimprec), Beta * (1 + betaimprec))
                eta = random.uniform(Eta * (1 - etaimprec), Eta * (1 + etaimprec))
                lbda = random.uniform(Lbda * (1 - lbdaimprec), Lbda * (1 + lbdaimprec))
                cp = random.uniform(Cp * (1 - cpimprec), Cp * (1 + cpimprec))
                cv = random.uniform(Cv * (1 - cvimprec), Cv * (1 + cvimprec))
                co = random.uniform(Co * (1 - coimprec), Co * (1 + coimprec))
                cf = random.uniform(Cf * (1 - cfimprec), Cf * (1 + cfimprec))
                cw = random.uniform(Cw * (1 - cwimprec), Cw * (1 + cwimprec))
                p = random.uniform(P * (1 - etaimprec), P * (1 + etaimprec))
                cr = objetivo(y)  # Corrigindo a passagem de y como argumento
                Lista_test.append(cr)
            
            # Exibindo média e desvio padrão
            st.write("Média:", sum(Lista_test) / len(Lista_test))
            st.write('Desvio Padrão:', np.std(Lista_test))
            
            # Filtrar apenas os números reais em Lista_test
            Lista_test_numeric = [x for x in Lista_test if isinstance(x, (int, float))]
            
            # Criar box-plot apenas se houver elementos na Lista_test_numeric
            if Lista_test_numeric:
                st.write('Box-Plot da Taxa de Custo')
                fig, ax = plt.subplots()
                ax.boxplot(Lista_test_numeric)
                st.pyplot(fig)
            else:
                st.write('Não há dados válidos para criar o box-plot.')

            
    if choice == menu[1]:
        st.header(menu[1])
        st.write('''Fazer o texto para colocar aqui''')

    if choice == menu[2]:
        st.header(menu[2])
        st.write('''The Research Group on Risk and Decision Analysis in Operations and Maintenance was created in 2012 
        in order to bring together different researchers who work in the following areas: risk, maintenance and 
        operation modelling. Learn more about it through our website.''')
        st.markdown('[Click here to be redirected to our website](http://random.org.br/en/)', False)

if __name__ == "__main__":
    main()

