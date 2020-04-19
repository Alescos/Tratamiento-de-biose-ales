#%%ñibrerias
import sys
#Qfiledialog es una ventana para abrir yu gfuardar archivos
#Qvbox es un organizador de widget en la ventana, este en particular los apila en vertcal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog,QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIntValidator

from matplotlib.figure import Figure

from PyQt5.uic import loadUi

from numpy import arange, sin, pi
#contenido para graficos de matplotlib
from matplotlib.backends. backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import scipy.io as sio
import numpy as np
from Modelo import Biosenal
# clase con el lienzo (canvas=lienzo) para mostrar en la interfaz los graficos matplotlib, el canvas mete la grafica dentro de la interfaz
class MyGraphCanvas(FigureCanvas):
    #constructor
    def __init__(self, parent= None,width=5, height=4, dpi=100):
        
        #se crea un objeto figura
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #el axes en donde va a estar mi grafico debe estar en mi figura
        self.axes = self.fig.add_subplot(111)
        
        #llamo al metodo para crear el primer grafico
        self.compute_initial_figure()
        
        #se inicializa la clase FigureCanvas con el objeto fig
        FigureCanvas.__init__(self,self.fig)
        
    #este metodo me grafica al senal senoidal que yo veo al principio, mas no senales
    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t,s)
    #hay que crear un metodo para graficar lo que quiera
    def graficar_gatos(self,datos):
        #primero se necesita limpiar la grafica anterior
        self.axes.clear()
        #ingresamos los datos a graficar
        self.axes.plot(datos)
        #y lo graficamos
        print("datos")
        print(datos)
        #voy a graficar en un mismo plano varias senales que no quecden superpuestas cuando uso plot me pone las graficas en un mismo grafico
        for c in range(datos.shape[0]):
            self.axes.plot(datos[c,:]+c*10)
        self.axes.set_xlabel("muestras")
        self.axes.set_ylabel("voltaje (uV)")
        #self.axes.set
        #ordenamos que dibuje
        self.axes.figure.canvas.draw()
    def graficar_canal(self,senal,n):
        #self.axes.plot(canal)
        if n <= (len(senal)-1):
            self.axes.clear()
            
            if senal.ndim>1:
                self.axes.plot(senal[n,:])
            else:
               self.axes.plot(senal)
            
        #self.axes.set
        #ordenamos que dibuje
        self.axes.set_xlabel("muestras")
        self.axes.set_ylabel("voltaje (uV)")
        self.axes.figure.canvas.draw()
        
 
            
            
#%%
        #es una clase que yop defino para crear los intefaces graficos
class InterfazGrafico(QMainWindow):
    #condtructor
    def __init__(self):
        #siempre va
        super(InterfazGrafico,self).__init__()
        #se carga el diseno
        loadUi ('anadir_grafico.ui',self)
        #se llama la rutina donde configuramos la interfaz
        self.setup()
        #se muestra la interfaz
        self.show()
    def setup(self):
        #los layout permiten organizar widgets en un contenedor
        #esta clase permite añadir widget uno encima del otro (vertical)
        layout = QVBoxLayout()
        #se añade el organizador al campo grafico
        self.campo_grafico.setLayout(layout)
        #se crea un objeto para manejo de graficos
        self.__sc = MyGraphCanvas(self.campo_grafico, width=5, height=4, dpi=100)
        #se añade el campo de graficos
        layout.addWidget(self.__sc)
        
        layout2 = QVBoxLayout()
        #se añade el organizador al campo grafico
        self.campo_grafico2.setLayout(layout2)
        #se crea un objeto para manejo de graficos
        self.__sc2 = MyGraphCanvas(self.campo_grafico2, width=5, height=4, dpi=100)
        #se añade el campo de graficos
        layout2.addWidget(self.__sc2)
        
        #se organizan las señales 
        self.boton_cargar.clicked.connect(self.cargar_senal)
        self.boton_adelante.clicked.connect(self.adelante_senal)
        self.boton_atras.clicked.connect(self.atrasar_senal)
        self.boton_aumentar.clicked.connect(self.aumentar_senal)
        self.boton_disminuir.clicked.connect(self.disminuir_senal)
        self.cargar_canal.clicked.connect(self.escoger_canal)
        self.cargar_canal2.clicked.connect(self.graficar_filtrada)
        self.guarda_filtrada.clicked.connect(self.guardar_filtrada)
        self.cargar_tiempo.clicked.connect(self.graficar_tiempo)
        
        #hay botones que no deberian estar habilitados si no he cargado la senal
        self.cargar_tiempo.setEnabled(False)
        self.canal_text.setEnabled(True)
        self.boton_adelante.setEnabled(False)
        self.boton_atras.setEnabled(False)
        self.boton_aumentar.setEnabled(False)
        self.boton_disminuir.setEnabled(False)
        self.nivel_filtrado.setEnabled(False)
        self.forma_filtrado.setEnabled(False)
        self.tipo_umbral.setEnabled(False)
        #cuando cargue la senal debo volver a habilitarlos
        
        
    def asignar_Controlador(self,controlador):
        self.__coordinador=controlador
    def adelante_senal(self):
        self.__x_min=self.__x_min+2000
        self.__x_max=self.__x_max+2000
        self.__sc.graficar_gatos(self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max))
    def atrasar_senal(self):
        #que se salga de la rutina si no puede atrazar
        if self.__x_min<2000:
            return
        self.__x_min=self.__x_min-2000
        self.__x_max=self.__x_max-2000
        self.__sc.graficar_gatos(self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max))
    def aumentar_senal(self):
        #en realidad solo necesito limites cuando tengo que extraerlos, pero si los 
        #extraigo por fuera mi funcion de grafico puede leer los valores
        self.__sc.graficar_gatos(self.__coordinador.escalarSenal(self.__x_min,self.__x_max,2))
    def disminuir_senal(self):
        self.__sc.graficar_gatos(self.__coordinador.escalarSenal(self.__x_min,self.__x_max,0.5))
    def escoger_canal(self):
    
        n = int(self.canal_text.text())
        self.__sc.graficar_canal(self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max),n)

    def graficar_tiempo(self): 
        #'''
       # Esta funcion permite obtener un segmento cualquiera de la se�al de entrada
       # y de la senal filtrada. Recibe los valores de tiempo minio y maximo.
        #'''
        
        time_min = int(self.tiempo_min.text())
        time_max = int(self.tiempo_max.text())
        
        canal = int(self.canal_filtro.text())
        senal=self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max)
        nivel = int(self.nivel_filtrado.currentIndex())
        forma = int(self.forma_filtrado.currentIndex())
        umbral = int(self.tipo_umbral.currentIndex())

        self.__x_min = time_min
        self.__x_max = time_max
        self.__sc.graficar_canal(self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max),canal)
        self.__sc2.graficar_canal((self.__coordinador.filtrarSenal(senal[canal,:],nivel,forma,umbral)),0)
        
    
        
    def cargar_senal(self):
        #se abre el cuadro de dialogo para cargar
        #* son archivos .mat
        archivo_cargado, _ = QFileDialog.getOpenFileName(self, "Abrir señal","","Todos los archivos (*);;Archivos mat (*.mat)*")
        if archivo_cargado != "":
            print(archivo_cargado)
            #la senal carga exitosamente entonces habilito los botones
            data = sio.loadmat(archivo_cargado)
            data = data["data"]
            #volver continuos los datos
            sensores,puntos,ensayos=data.shape
            senal_continua=np.reshape(data,(sensores,puntos*ensayos),order="F")
            #el coordinador recibe y guarda la senal en su propio .py, por eso no 
            #necesito una variable que lo guarde en el .py interfaz
            self.__coordinador.recibirDatosSenal(senal_continua)
            self.__x_min=0
            self.__x_max=2000
            #graficar utilizando el controlador
            self.__sc.graficar_gatos(self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max))
            self.boton_adelante.setEnabled(True)
            self.boton_atras.setEnabled(True)
            self.boton_aumentar.setEnabled(True)
            self.boton_disminuir.setEnabled(True)
            self.nivel_filtrado.setEnabled(True)
            self.forma_filtrado.setEnabled(True)
            self.tipo_umbral.setEnabled(True)    
            self.cargar_tiempo.setEnabled(True)

    def graficar_filtrada(self):
        #"""
        #Esta se�al grafica la se�al filtrada. Recibe el canal y la se�al de entrada
        #y retorna el grafico de la se�al filtrada en el canal deseado
        #"""
        
#        nivel_actual = 1
        canal = int(self.canal_filtro.text())
        senal=self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max)
        if canal<= (len(senal)-1):
            
            nivel = int(self.nivel_filtrado.currentIndex())
            forma = int(self.forma_filtrado.currentIndex())
            umbral = int(self.tipo_umbral.currentIndex())
            if (nivel == 0): 
                msg=QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setText("Informacion")
                msg.setWindowTitle('Advertencia!')
                msg.setInformativeText('Elija un nivel diferente')
                msg.show()
            if (forma == 0): 
                msg=QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setText("Informacion")
                msg.setWindowTitle('Advertencia!')
                msg.setInformativeText('Elija una forma diferente')
                msg.show()
            if (umbral == 0): 
                msg=QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setText("Informacion")
                msg.setWindowTitle('Advertencia!')
                msg.setInformativeText('Elija un umbral diferente')
                msg.show();

            self.__sc2.graficar_canal((self.__coordinador.filtrarSenal(senal[canal,:],nivel,forma,umbral)),0)
        
        else:
            msg=QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setText("Informacion")
            msg.setWindowTitle('Advertencia!')
            msg.setInformativeText('El canal seleccionado no existe, elija otro.')
            msg.show()
    
    def guardar_filtrada(self):
        #"""
        #Esta funci�n guarda la se�al filtrada en un archivo .mat, haciendo uso 
        #de la funci�n savemat de Scipy
        #"""
        
        canal = int(self.canal_filtro.text())
        senal = self.__coordinador.devolverDatosSenal(self.__x_min,self.__x_max)
        nivel = int(self.nivel_filtrado.currentIndex())
        forma = int(self.forma_filtrado.currentIndex())
        umbral = int(self.tipo_umbral.currentIndex())
        sio.savemat('Senal_Filtrada.mat',{'data':self.__coordinador.filtrarSenal(senal[canal,:],nivel,forma,umbral)})   
            
            
        
            
#app=QApplication(sys.argv) 
#mi_ventana = InterfazGrafico() 
#sys.exit(app.exec_()) 
   