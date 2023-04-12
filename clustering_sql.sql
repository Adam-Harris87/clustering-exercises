use zillow;
SHOW tables;
describe predictions_2017;
describe properties_2017;
describe unique_properties;
select * from unique_properties;
select * from airconditioningtype;

SELECT *
FROM properties_2017
JOIN predictions_2017
	USING (parcelid)
LEFT JOIN airconditioningtype
	USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype
	USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype
	USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype
	USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype
	USING (propertylandusetypeid)
LEFT JOIN storytype
	USING (storytypeid)
LEFT JOIN typeconstructiontype
	USING (typeconstructiontypeid)
WHERE latitude IS NOT NULL
	AND longitude IS NOT NULL
    AND parcelid IN (
		SELECT parcelid FROM unique_properties)
;

use mall_customers;
SELECT * FROM customers;