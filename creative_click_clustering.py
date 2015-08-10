import bson
from datetime import date

from flask.ctx import AppContext

from pydash.app import app as pydash_app
from pydash.db import db
from cbanalytics.analytics_client import AnalyticsClient

from timeit import default_timer


def create_pydash_context():
    context = AppContext(pydash_app)
    context.push()
    ctx = pydash_app.test_request_context('/?next=http://example.com/')
    ctx.push()


def build_query(date):
    query_formated_date = date.strftime('%Y-%m-%d')
    # query = 'SELECT DISTINCT publisher_app FROM warehouse.daily_uber_aggr WHERE money_earned > 0 AND dt = "{date}" LIMIT 1'.format(date=query_formated_date)
    query = 'SELECT click_id, dt, asset_ratio, asset_size, ad_type, creative, device_id, publisher_app, event_location, sdk, country, model, model_id, creative_type, bid_type, bid_value FROM warehouse.clicks WHERE dt = "2015-07-25" limit 100'
    # query = ''
    print query
    return query

def get_click_data():
    analytics_client = AnalyticsClient()
    analytics_data = analytics_client.schedule_query(build_query(date.today()))

    return analytics_data


def update_db(ad_revenue_results, log=True):
    count = 1
    for record in ad_revenue_results:
        print record

        if log:
            print "{0}) Updated AppID: {1}".format(count, record['publisher_app'])

        db.apps.update(
            {'_id': bson.ObjectId(record['publisher_app'])},
            {"$addToSet": {"product_integrations": AD_REVENUE_FLAG}},
            upsert=False
        )
        count += 1

    if log:
        print "Updated {0} Apps with  'ad_revenue' flag into 'product_integrations' key/field ".format(count - 1)


def main():
    create_pydash_context()
    analytics_client = AnalyticsClient()

    tic = default_timer()
    click_results = get_click_data()
    toc = default_timer()

    for click in click_results:
        print click
        print click['model_id']

    print 'Total Query Time: ' + str(toc-tic)
    # update_db(ad_revenue_results)

if __name__ == "__main__":
    main()
