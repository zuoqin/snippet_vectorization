package JavaExtractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import org.kohsuke.args4j.CmdLineException;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.FeaturesEntities.SnippetRelation;

public class App {
	private static CommandLineValues s_CommandLineValues;

	public static void main(String[] args) {
		try {
			s_CommandLineValues = new CommandLineValues(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
			return;
		}

		if (s_CommandLineValues.NoHash) {
			SnippetRelation.setNoHash();
		}

		if (s_CommandLineValues.File != null) {
			ExtractFeatures ExtractFeatures = new ExtractFeatures(s_CommandLineValues,
					s_CommandLineValues.File.toPath());
			ExtractFeatures.processFile();
		} else if (s_CommandLineValues.Dir != null) {
			extractDir();
		}
	}

	private static void extractDir() {
		ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
		LinkedList<ExtractFeatures> tasks = new LinkedList<>();
		try {
			Files.walk(Paths.get(s_CommandLineValues.Dir)).filter(Files::isRegularFile)
					.filter(p -> p.toString().toLowerCase().endsWith(".java")).forEach(f -> {
						ExtractFeatures task = new ExtractFeatures(s_CommandLineValues, f);
						tasks.add(task);
					});
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		try {
			executor.invokeAll(tasks);
		} catch (InterruptedException e) {
			e.printStackTrace();
		} finally {
			executor.shutdown();
		}
	}
}
