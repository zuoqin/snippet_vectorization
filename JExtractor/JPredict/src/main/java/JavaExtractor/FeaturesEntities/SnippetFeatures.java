package JavaExtractor.FeaturesEntities;

import java.util.ArrayList;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class SnippetFeatures {
	private String name;

	private ArrayList<SnippetRelation> features = new ArrayList<>();

	public SnippetFeatures(String name) {
		this.name = name;
	}

	@SuppressWarnings("StringBufferReplaceableByString")
	@Override
	public String toString() {
		StringBuilder stringBuilder = new StringBuilder();
		stringBuilder.append(name).append(" ");
		stringBuilder.append(features.stream().map(SnippetRelation::toString).collect(Collectors.joining(" ")));

		return stringBuilder.toString();
	}

	public void addFeature(Property source, String path, Property target) {
		SnippetRelation newRelation = new SnippetRelation(source, target, path);
		features.add(newRelation);
	}

	@JsonIgnore
	public boolean isEmpty() {
		return features.isEmpty();
	}

	public void deleteAllPaths() {
		features.clear();
	}

	public String getName() {
		return name;
	}

	public ArrayList<SnippetRelation> getFeatures() {
		return features;
	}

}
