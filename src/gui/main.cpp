#include <QApplication>

#include "MainWindow.hpp"

int main(int argc, char *argv[])
 {
     QApplication app(argc, argv);
	 MainWindow win; 
     //Client client;
     //client.show();
	 win.show();
     //return client.exec();
	 return app.exec();
 }
