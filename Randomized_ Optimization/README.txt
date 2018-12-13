Any source code I have written was all done in Java 8 (and one Python 3.6 file)
ABAGAIL, an open-source Java machine learning library - https://github.com/pushkar/ABAGAIL

Installation (pulled from original ABAGAIL repo):
1. Install Java 8 SDK from here http://www.oracle.com/technetwork/java/javase/downloads/index.html
2. Install Ant http://ant.apache.org/
3. Clone or download source files from Git
4. Go with command line to where the build.xml file is and run: ant (note: the ant executable should be in your path somehow if you installed ant correctly.. so will java and javac)
5. Now run your scripts

How to use:
1. Enter the ABAGAIL folder
2. Open a Terminal, PowerShell, or other command line utility
3. Type "ant" (this will compile all the source files; make sure ant is installed!)
4. Select a test file to run from src/opt/test/
5. The dataset for first assignment is "cleandata_1.csv". Run this command in your command line from the main directory: "java -cp ABAGAIL.jar opt.test.<TestName>" without quotes, and replacing <TestName> with your test of choice
6. Save your results in a txt file by command:"ABAGAIL.jar opt.test.ProteinSolubilityTest<TestName> &> <TestName>.txt"


