#import os
#os.environ['DYLD_LIBRARY_PATH'] = '/Library/PostgreSQL/9.5/lib'
import psycopg2
import time
import itertools

# Dictionary of database connections. Key is the database name (only within python relevant). Value is List of: [host, user, password, database, port]
databaseSetups = {
    
}


class Database:

    def __init__(self, setup):
        '''
            @param setup: string. Database Setup. Must match a key from the databaseSetups dictionary (see above)
            Creates an connection to database
        '''
        self.setup = setup
        self.db = self._getConnection()

    def _getConnection(self):
        '''
            Returns database connection. Function checks if database connection is still alive and if this is not the case it automatically reconnects to database
            @return: MySQLdb connection object
        '''
        try:
            # check if database connection is still alive
            db = self.db
            cursor = db.cursor()
            cursor.execute("SELECT VERSION()")
            results = cursor.fetchone()

            # if result is not empty connection is alive
            if results:
                return db
                # create new connection
            else:
                connectionDetails = databaseSetups[self.setup]
                self.db = psycopg2.connect(host = connectionDetails[0], user = connectionDetails[1], password = connectionDetails[2], database = connectionDetails[3], port = connectionDetails[4])
                return self.db

        except:
            # if an exception occurs during checking if database connection is still alive create a new connection; if it's not possible to connect to Datbase Server wait 120 sec and try to connect again
            reConnect = True
            while reConnect is True:
                try:
                    connectionDetails = databaseSetups[self.setup]
                    self.db = psycopg2.connect(host = connectionDetails[0], user = connectionDetails[1], password = connectionDetails[2], database = connectionDetails[3], port = connectionDetails[4])
                    reConnect = False
                except:
                    time.sleep(120)
            return self.db

    def closeConnection(self):
        '''
            @summary: Closes connection to database
            @author: JScelle
            @version: 10.06.2013
        '''
        try:
            self.db.close()
        except:
            # if closing of connection fails ignore it
            pass

    def getValuesFromTable(self, sqlQueryStatement):
        '''
            gets records from table
            @param sqlQueryStatement: String, sql query statement e.g. "SELECT * FROM employee WHERE country = 'DE'"
            @return: tuple of tuples
        '''
        # get database connection from runTimeController
        try:
            db = self.db
        except Exception as exc:
            print("EXCEPTION: Module: Database, Function: getValuesFromTable, Error: Could not get database connection. ", str(exc.args))
            return ((),)

        #excecute sql statement
        try:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            # Execute the SQL command
            cursor.execute(sqlQueryStatement)
            # Fetch all the rows in a list of lists.
            results = cursor.fetchall()
            db.commit()
            return results
        except Exception as exc:
            # Rollback in case there is any error
            db.rollback()
            print("EXCEPTION: Module: Database, Function: getValuesFromTable, Error: Could not execute query. ", str(exc.args))
            return ((),)

    def runQuery(self, sqlStatement):
        '''
            Executes any sql statement. For getting data from the database, please use the getValuesFromTable-method
            @param sqlStatement: string. any sql statement e.g. Update, Delete
            @return: Returns True if everything went good.
        '''
        # get database connection from runTimeController
        try:
            db = self.db
        except Exception as exc:
            print("EXCEPTION: Module: Database, Function: runQuery, Error: Could not get database connection. ", str(exc.args))
            return False

        #excecute sql statement
        try:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            # Execute the SQL command
            cursor.execute(sqlStatement)
            # Commit your changes in the Settings.database
            db.commit()
            return True
        except Exception as exc:
            # Rollback in case there is any error
            db.rollback()
            print("EXCEPTION: Module: Database, Function: runQuery, Error: Could not execute query. ", str(exc.args))
            return False

    def runMultipleQueries(self, sqlStatements):
        """
            Executes multiple SQL-statements as one transaction
            @param sqlStatements: List of strings. Each String is one SQL-statement
            @return: Returns True if everything went good.
        """
        # get database connection from runTimeController
        try:
            db = self.db
        except Exception as exc:
            print("EXCEPTION: Module: Database, Function: runMultipleQueries, Error: Could not get database connection. ", str(exc.args))
            return False

        # excecute sql statement
        try:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()
            for sqlStatement in sqlStatements:
                # Execute the SQL command
                cursor.execute(sqlStatement)
            # Commit your changes in the Settings.database
            db.commit()
            return True
        except Exception as exc:
            # Rollback in case there is any error
            db.rollback()
            print("EXCEPTION: Module: Database, Function: runMultipleQueries, Error: One of the queries could not be executed. ", str(exc.args))
            return False


    def insertValuesIntoTable(self, tableName, fieldList, valueList):
        """
            calls _executeInsertValuesIntoTable which inserts values into table
            @param tableName: String, name of table
            @param fieldList: List of string, list of field names  e.g ,["firstName", "lastName"]
            @param valueList: List of list, e.g [["Sebastian", "Stuetz"],["Ingo", "Tschach"]]
            @return: Returns True if everything went good.
        """
        try:
            #define INSERT INTO part of sql query
            insertIntoSqlPart = "INSERT INTO " + tableName + " (" + ",".join(fieldList) + ") VALUES"
            #define ON DUPLICATE KEY part of sql query
            onDuplicateKeyPart = " ON DUPLICATE KEY UPDATE "
            for field in fieldList:
                #replace old values with new values
                onDuplicateKeyPart += str(field) + " = VALUES(" + str(field) + "), "
            onDuplicateKeyPart = onDuplicateKeyPart[0:(len(onDuplicateKeyPart) - 2)]

            return self._executeInsertValuesIntoTable(fieldList, valueList, insertIntoSqlPart, onDuplicateKeyPart)

        except Exception as exc:
            print("EXCEPTION: Module: Database, Function: insertValuesIntoTable, Error: Error when building the query frame. ", str(exc.args))
            return False


    def _executeInsertValuesIntoTable(self, fieldList, valueList, insertIntoSqlPart, onDuplicateKeyPart):
        '''
            inserts values into table
            @param fieldList: List, list of field names  e.g ,["firstName", "lastName"]
            @param valueList: List of list, e.g [["Sebastian", "Stuetz"],["Ingo", "Tschach"]]
            @return: Returns True if everything went good.
        '''
        # get database connection from runTimeController
        try:
            db = self.db
        except Exception as exc:
            print("EXCEPTION: Module: Database, Function: _executeInsertValuesIntoTable, Error: Could not get database connection. ", str(exc.args))
            return False

        try:
            # prepare a cursor object using cursor() method
            cursor = db.cursor()

            #Insert data
            #divide valueList into chunks to avoid that too big packets are send to sql server
            maxPackets = 5000
            for i in range(0, len(valueList), maxPackets):
                chunk = valueList[i:i + maxPackets]
                #define data part of sql query (string consisting of "%s" for each field for each row)
                dataSqlPart = ",".join(["(" + ",".join(["%s"] * len(fieldList)) + ")"] * len(chunk))
                #put sql parts together
                sql = insertIntoSqlPart + dataSqlPart + onDuplicateKeyPart
                #flatten data list
                valueListChunk = tuple(itertools.chain(*chunk))
                #Insert values into DB
                cursor.execute(sql, valueListChunk)

            # Commit your changes in the Settings.database
            db.commit()

            return True
        except Exception as exc:
            # Rollback in case there is any error
            db.rollback()
            print("EXCEPTION: Module: Database, Function: _executeInsertValuesIntoTable, Error: Could not insert values. ", str(exc.args))
            return False

