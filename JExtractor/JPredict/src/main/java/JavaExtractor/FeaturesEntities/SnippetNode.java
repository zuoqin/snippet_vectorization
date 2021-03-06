package JavaExtractor.FeaturesEntities;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import JavaExtractor.Common.Common;

public class SnippetNode {
	public int Id;
	public String Type;
	public String Name;
	public boolean IsMethodDeclarationName;
	
	public SnippetNode(String name) {
		Name = name;
		try {
			Name = URLEncoder.encode(name, Common.UTF8);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
}
